import os
import sys
import math
import argparse
import torch
import torch._dynamo
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.upsampling import Upsample2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.embeddings import Timesteps

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        diff = torch.max(torch.abs(out.cpu() - cpu_out)).item()
        print(f"Max abs diff: {diff}")
        exit(1)

@torch.no_grad()
def test_unet_conditional(
    device,
    model_id="runwayml/stable-diffusion-v1-5",
    batch=1,
    dtype="float32",
    rtol=1e-4,
    atol=1e-4,
    prompt="a cat in a hat",
):
    from diffusers import DiffusionPipeline

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading pipeline: {model_id} (dtype={torch_dtype})")
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.to("cpu")

    unet = pipe.unet.eval()
    in_ch = unet.config.in_channels
    latent_sz = getattr(unet.config, "sample_size", 64)
    cross_dim = getattr(unet.config, "cross_attention_dim", None)

    g = torch.Generator().manual_seed(0)
    latents = torch.randn(batch, in_ch, latent_sz, latent_sz, generator=g, dtype=torch_dtype)
    timestep = torch.tensor(999, dtype=torch.float32)

    enc_states_cpu = None
    if hasattr(pipe, "tokenizer") and hasattr(pipe, "text_encoder") and cross_dim is not None:
        try:
            tokens = pipe.tokenizer(
                [prompt] * batch,
                padding="max_length",
                max_length=getattr(pipe.tokenizer, "model_max_length", 77),
                truncation=True,
                return_tensors="pt",
            )
            text_out = pipe.text_encoder(input_ids=tokens.input_ids).last_hidden_state  # [B, T, D]
            if text_out.shape[-1] != cross_dim:
                print(f"Warning: text_encoder dim {text_out.shape[-1]} != cross_attn dim {cross_dim}. Fallback to random.")
                raise RuntimeError("cross-dim mismatch")
            enc_states_cpu = text_out.to(dtype=torch_dtype)
        except Exception as e:
            print(f"Text encoder unavailable or mismatch: {e}. Fallback to random encoder states.")
    if enc_states_cpu is None:
        if cross_dim is None:
            enc_states_cpu = None
        else:
            seq_len = 77
            enc_states_cpu = torch.randn(batch, seq_len, cross_dim, generator=g, dtype=torch_dtype)

    latents_dev = latents.to(device)
    timestep_dev = timestep.to(device)
    if enc_states_cpu is not None:
        enc_states_dev = enc_states_cpu.to(device)
    else:
        enc_states_dev = None

    print("Compiling UNet with torch.compile(...)")
    unet_dev = unet.to(device)
    unet_compiled = torch.compile(unet_dev, dynamic=False)

    # Forward (device)
    with torch.no_grad():
        if enc_states_dev is None:
            out_dev = unet_compiled(latents_dev, timestep_dev).sample
        else:
            out_dev = unet_compiled(latents_dev, timestep_dev, encoder_hidden_states=enc_states_dev).sample

        unet_cpu = unet.to("cpu")
        if enc_states_cpu is None:
            out_cpu = unet_cpu(latents.cpu(), timestep).sample
        else:
            out_cpu = unet_cpu(latents.cpu(), timestep, encoder_hidden_states=enc_states_cpu).sample

    test_result(f"UNet({model_id}) forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu() - out_cpu)).item())
    print("UNet Simulation Done")

def test_unet_mid_block2d_cross_attn(
    device,
    in_channels=320,
    temb_channels=320,
    cross_attention_dim=768,
    batch=1,
    height=32,
    width=32,
    rtol=1e-4,
    atol=1e-4,
    num_layers=1,
    num_attention_heads=8,
    dual_cross_attention=False,
):
    print(f"Testing UNetMidBlock2DCrossAttn on device: {device}")

    cpu_block = UNetMidBlock2DCrossAttn(
        in_channels=in_channels,
        temb_channels=temb_channels,
        num_layers=num_layers,
        cross_attention_dim=cross_attention_dim,
        num_attention_heads=num_attention_heads,
        dual_cross_attention=dual_cross_attention,
    ).to("cpu").eval()

    g = torch.Generator().manual_seed(0)
    hidden_states_cpu = torch.randn(batch, in_channels, height, width, generator=g)
    temb_cpu = torch.randn(batch, temb_channels, generator=g)
    encoder_hidden_states_cpu = torch.randn(batch, 77, cross_attention_dim, generator=g)

    with torch.no_grad():
        cpu_out = cpu_block(
            hidden_states=hidden_states_cpu,
            temb=temb_cpu,
            encoder_hidden_states=encoder_hidden_states_cpu,
        )

    dev_block = cpu_block.to(device).eval()
    dev_block = torch.compile(dev_block, dynamic=False)

    hidden_states_dev = hidden_states_cpu.to(device)
    temb_dev = temb_cpu.to(device)
    encoder_hidden_states_dev = encoder_hidden_states_cpu.to(device)

    with torch.no_grad():
        dev_out = dev_block(
            hidden_states=hidden_states_dev,
            temb=temb_dev,
            encoder_hidden_states=encoder_hidden_states_dev,
        )

    test_result("UNetMidBlock2DCrossAttn", dev_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(dev_out.cpu() - cpu_out)).item())
    print("UNetMidBlock2DCrossAttn simulation done.")

def test_cross_attn_up_block2d(
    device,
    in_channels=320,
    out_channels=320,
    prev_output_channel=320,
    temb_channels=1280,
    cross_attention_dim=768,
    batch=1,
    height=32,
    width=32,
    rtol=1e-4,
    atol=1e-4,
    num_layers=1,
    num_attention_heads=8,
    dual_cross_attention=False,
):
    print(f"Testing CrossAttnUpBlock2D on device: {device}")

    cpu_block = CrossAttnUpBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        prev_output_channel=prev_output_channel,
        temb_channels=temb_channels,
        num_layers=num_layers,
        cross_attention_dim=cross_attention_dim,
        num_attention_heads=num_attention_heads,
        dual_cross_attention=dual_cross_attention,
        # add_upsample=add_upsample,
    ).to("cpu").eval()

    g = torch.Generator().manual_seed(0)
    hidden_states_cpu = torch.randn(batch, in_channels, height, width, generator=g)
    temb_cpu = torch.randn(batch, temb_channels, generator=g)
    encoder_hidden_states_cpu = torch.randn(batch, 77, cross_attention_dim, generator=g)

    res_hidden_states_tuple_cpu = tuple(
        torch.randn(batch, prev_output_channel, height, width, generator=g) for _ in range(num_layers)
    )

    with torch.no_grad():
        cpu_out = cpu_block(
            hidden_states=hidden_states_cpu,
            res_hidden_states_tuple=res_hidden_states_tuple_cpu,
            temb=temb_cpu,
            encoder_hidden_states=encoder_hidden_states_cpu,
        )

    dev_block = cpu_block.to(device).eval()
    dev_block = torch.compile(dev_block, dynamic=False)

    hidden_states_dev = hidden_states_cpu.to(device)
    temb_dev = temb_cpu.to(device)
    encoder_hidden_states_dev = encoder_hidden_states_cpu.to(device)
    res_hidden_states_tuple_dev = tuple(t.to(device) for t in res_hidden_states_tuple_cpu)

    with torch.no_grad():
        dev_out = dev_block(
            hidden_states=hidden_states_dev,
            res_hidden_states_tuple=res_hidden_states_tuple_dev,
            temb=temb_dev,
            encoder_hidden_states=encoder_hidden_states_dev,
        )

    test_result("CrossAttnUpBlock2D", dev_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(dev_out.cpu() - cpu_out)).item())
    print("CrossAttnUpBlock2D simulation done.")

def test_unet2d_condition_model(
    device,
    batch=1,
    in_channels=4,
    out_channels=4,
    sample_size=32,
    cross_attention_dim=[768, 768],
    seq_len=77,
    block_out_channels=(64, 64),
    layers_per_block=[1, 1],
    attention_head_dim=(8, 8),
    rtol=1e-4,
    atol=1e-4,
    stride=None,
):
    down_block_types = ("CrossAttnDownBlock2D", "DownBlock2D")
    up_block_types   = ("UpBlock2D", "CrossAttnUpBlock2D")

    unet_cpu = UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim,
    ).to("cpu").eval()

    g = torch.Generator().manual_seed(0)

    if stride is not None:
        x_cpu = torch.empty_strided([batch, in_channels, sample_size, sample_size], stride).normal_(generator=g)
    else:
        x_cpu = torch.randn(batch, in_channels, sample_size, sample_size, generator=g)

    t_cpu = torch.randint(low=0, high=1000, size=(batch,), generator=g, dtype=torch.long)
    encoder_hidden_states_cpu = torch.randn(batch, seq_len, cross_attention_dim[0], generator=g)

    # CPU result
    with torch.no_grad():
        y_cpu = unet_cpu(
            sample=x_cpu,
            timestep=t_cpu,
            encoder_hidden_states=encoder_hidden_states_cpu,
        ).sample  # UNet2DConditionOutput.sample (Tensor)

    # Device + torch.compile
    unet_dev = unet_cpu.to(device).eval()
    unet_dev = torch.compile(unet_dev, dynamic=False)

    x_dev = x_cpu.to(device)
    t_dev = t_cpu.to(device)
    encoder_hidden_states_dev = encoder_hidden_states_cpu.to(device)

    with torch.no_grad():
        y_dev = unet_dev(
            sample=x_dev,
            timestep=t_dev,
            encoder_hidden_states=encoder_hidden_states_dev,
        ).sample

    for idx, (cpu, dev) in enumerate(zip(y_cpu, y_dev)):
        test_result(f"[{idx}] UNet2DConditionModel", dev.cpu(), cpu, rtol=rtol, atol=atol)
        max_diff = torch.max(torch.abs(dev.detach().cpu() - cpu)).item()
        print("Max diff >", max_diff)
    print("UNet2DConditionModel simulation done.")

def test_cross_attn_down_block2d(
    device,
    in_channels=320,
    out_channels=320,
    temb_channels=1280,
    cross_attention_dim=768,
    batch=1,
    height=32,
    width=32,
    rtol=1e-4,
    atol=1e-4,
    num_layers=1,
    num_attention_heads=8,
    dual_cross_attention=False
):
    print(f"Testing CrossAttnDownBlock2D on device: {device}")

    # 1. Initialize the module on CPU
    cpu_block = CrossAttnDownBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        num_layers=num_layers,
        cross_attention_dim=cross_attention_dim,
        num_attention_heads=num_attention_heads,
        dual_cross_attention=dual_cross_attention
    ).to("cpu").eval()

    # 2. Create synthetic inputs on CPU
    g = torch.Generator().manual_seed(0)
    hidden_states_cpu = torch.randn(batch, in_channels, height, width, generator=g)
    temb_cpu = torch.randn(batch, temb_channels, generator=g)
    encoder_hidden_states_cpu = torch.randn(batch, 77, cross_attention_dim, generator=g)

    # 3. Get the output from the CPU module
    with torch.no_grad():
        cpu_out, _ = cpu_block(
            hidden_states=hidden_states_cpu,
            temb=temb_cpu,
            encoder_hidden_states=encoder_hidden_states_cpu,
        )

    # 4. Initialize the module on the custom device
    device_block = cpu_block.to(device).eval()
    device_block = torch.compile(device_block, dynamic=False)

    # 5. Move inputs to the custom device
    hidden_states_dev = hidden_states_cpu.to(device)
    temb_dev = temb_cpu.to(device)
    encoder_hidden_states_dev = encoder_hidden_states_cpu.to(device)

    # 6. Get the output from the custom device module
    with torch.no_grad():
        dev_out, _ = device_block(
            hidden_states=hidden_states_dev,
            temb=temb_dev,
            encoder_hidden_states=encoder_hidden_states_dev,
        )

    # 7. Compare the results
    test_result("CrossAttnDownBlock2D", dev_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(dev_out.cpu() - cpu_out)).item())
    print("CrossAttnDownBlock2D simulation done.")

def test_resnetblock2d(
    device,
    batch=1,
    in_channels=320,
    out_channels=320,
    height=32,
    width=32,
    temb_channels=128,
    resnet_eps=1e-5,
    resnet_groups=32,
    dropout=0.0,
    resnet_time_scale_shift="default",   # e.g., "default" | "scale_shift"
    resnet_act_fn="swish",
    output_scale_factor=1.0,
    resnet_pre_norm=True,
    rtol=1e-4,
    atol=1e-4,
    stride=None,
):
    print(f"Testing ResnetBlock2D(down=True) on device: {device}")

    g = torch.Generator().manual_seed(0)
    cpu_blk = ResnetBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        eps=resnet_eps,
        groups=resnet_groups,
        dropout=dropout,
        time_embedding_norm=resnet_time_scale_shift,
        non_linearity=resnet_act_fn,
        output_scale_factor=output_scale_factor,
        pre_norm=resnet_pre_norm
    ).to("cpu").eval()

    if stride is not None:
        x_cpu = torch.empty_strided([batch, in_channels, height, width], stride).normal_()
    else:
        x_cpu = torch.randn(batch, in_channels, height, width, generator=g)

    temb_cpu = torch.randn(batch, temb_channels, generator=g)

    with torch.no_grad():
        y_cpu = cpu_blk(x_cpu, temb=temb_cpu)

    dev_blk = cpu_blk.to(device).eval()
    dev_blk = torch.compile(dev_blk, dynamic=False)

    x_dev = x_cpu.to(device)
    temb_dev = temb_cpu.to(device)

    with torch.no_grad():
        y_dev = dev_blk(x_dev, temb=temb_dev)

    try:
        test_result("ResnetBlock2D(down=True)", y_dev, y_cpu, rtol=rtol, atol=atol)
    except NameError:
        # fallback: PyTorch의 기본 엄밀 비교
        torch.testing.assert_close(y_dev.cpu(), y_cpu, rtol=rtol, atol=atol)
        print("ResnetBlock2D(down=True) close-check passed.")

    max_diff = torch.max(torch.abs(y_dev.cpu() - y_cpu)).item()
    print("Max diff >", max_diff)
    print("ResnetBlock2D simulation done.")

def test_groupnorm(
    device,
    batch=1,
    channels=320,
    height=32,
    width=32,
    num_groups=32,
    eps=1e-5,
    rtol=1e-4,
    atol=1e-4,
    stride=None
):
    print(f"Testing GroupNorm on device: {device}")

    # 1. Initialize the module on CPU
    cpu_norm = torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=channels,
        eps=eps,
        affine=True
    ).to("cpu").eval()

    # 2. Create synthetic inputs on CPU
    g = torch.Generator().manual_seed(0)
    if stride is not None:
        input_cpu = torch.empty_strided([batch, channels, height, width], stride)
        input_cpu = input_cpu.normal_()
    else:
        input_cpu = torch.randn(batch, channels, height, width, generator=g)

    # 3. Get the output from the CPU module
    with torch.no_grad():
        cpu_out = cpu_norm(input_cpu)

    # 4. Initialize the module on the custom device
    device_norm = torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=channels,
        eps=eps,
        affine=True
    ).to(device).eval()
    device_norm = torch.compile(device_norm, dynamic=False)

    # Copy the weights from the CPU module to ensure they are identical
    device_norm.weight.data.copy_(cpu_norm.weight.data)
    device_norm.bias.data.copy_(cpu_norm.bias.data)

    # 5. Move inputs to the custom device
    input_dev = input_cpu.to(device)

    # 6. Get the output from the custom device module
    with torch.no_grad():
        dev_out = device_norm(input_dev)

    # 7. Compare the results
    test_result("GroupNorm", dev_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(dev_out.cpu() - cpu_out)).item())
    print("GroupNorm simulation done.")

def test_upsample2d(
    device,
    batch=1,
    channels=320,
    height=32,
    width=32,
    rtol=1e-4,
    atol=1e-4,
    use_conv=True,
    use_conv_transpose=False,
    out_channels=320,
    name="conv",
    kernel_size=None,
    padding=1,
    norm_type=None,
    eps=None,
    elementwise_affine=None,
    bias=True,
    interpolate=True,
    stride=None,
):
    cpu_block = Upsample2D(
        channels=channels,
        use_conv=use_conv,
        use_conv_transpose=use_conv_transpose,
        out_channels=out_channels,
        name=name,
        kernel_size=kernel_size,
        padding=padding,
        norm_type=norm_type,
        eps=eps,
        elementwise_affine=elementwise_affine,
        bias=bias,
        interpolate=interpolate,
    ).to("cpu").eval()

    g = torch.Generator().manual_seed(0)
    if stride is not None:
        x_cpu = torch.empty_strided([batch, channels, height, width], stride).normal_(generator=g)
    else:
        x_cpu = torch.randn(batch, channels, height, width, generator=g)

    with torch.no_grad():
        y_cpu = cpu_block(x_cpu)

    dev_block = cpu_block.to(device).eval()
    dev_block = torch.compile(dev_block, dynamic=False)
    x_dev = x_cpu.to(device)

    with torch.no_grad():
        y_dev = dev_block(x_dev)

    test_result("Upsample2D", y_dev, y_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(y_dev.cpu() - y_cpu)).item())
    print("Upsample2D simulation done.")


def test_flip_sin_to_cos_embedding(
    device,
    batch=1,
    embedding_dim=256,
    rtol=1e-4,
    atol=1e-4,
):
    def create_embeddings(timesteps, embedding_dim, scale=1.0, flip_sin_to_cos=False):
        """
        Replicate the embedding creation logic from Timesteps class.
        """
        half_dim = embedding_dim // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = scale * emb

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # flip sine and cosine embeddings
        if flip_sin_to_cos:
            new_emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
            return emb, new_emb
        return emb, emb

    g = torch.Generator().manual_seed(0)
    timesteps_cpu = torch.randint(low=0, high=1000, size=(batch,), generator=g, dtype=torch.long)

    # Test with flip_sin_to_cos=True
    with torch.no_grad():
        emb_flip_cpu = create_embeddings(timesteps_cpu, embedding_dim, flip_sin_to_cos=True)

    # Move to device and test
    timesteps_dev = timesteps_cpu.to(device)
    @torch.compile(dynamic=False)
    def create_embeddings_compiled(timesteps, embedding_dim, scale=1.0, flip_sin_to_cos=False):
        return create_embeddings(timesteps, embedding_dim, scale, flip_sin_to_cos)

    with torch.no_grad():
        emb_flip_dev = create_embeddings_compiled(timesteps_dev, embedding_dim, flip_sin_to_cos=True)

    # Verify flip case
    test_result("Embedding (flip_sin_to_cos=True)", emb_flip_dev[0], emb_flip_cpu[0], rtol=rtol, atol=atol)
    print("Max diff (flip) >", torch.max(torch.abs(emb_flip_dev[0].cpu() - emb_flip_cpu[0])).item())
    test_result("Embedding (flip_sin_to_cos=True)", emb_flip_dev[1], emb_flip_cpu[1], rtol=rtol, atol=atol)
    print("Max diff (flip) >", torch.max(torch.abs(emb_flip_dev[1].cpu() - emb_flip_cpu[1])).item())


def test_timesteps(
    device,
    batch=1,
    num_channels=64,
    flip_sin_to_cos=True,
    downscale_freq_shift=1.0,
    rtol=1e-4,
    atol=1e-4,
):
    print(f"Testing Timesteps on device: {device}")

    cpu_timesteps = Timesteps(
        num_channels=num_channels,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
    ).to("cpu").eval()

    g = torch.Generator().manual_seed(0)
    timesteps_cpu = torch.randint(low=0, high=1000, size=(batch,), generator=g, dtype=torch.long)

    with torch.no_grad():
        cpu_out = cpu_timesteps(timesteps_cpu)

    dev_timesteps = cpu_timesteps.to(device).eval()
    dev_timesteps = torch.compile(dev_timesteps, dynamic=False)

    timesteps_dev = timesteps_cpu.to(device)
    with torch.no_grad():
        dev_out = dev_timesteps(timesteps_dev)

    test_result("Timesteps", dev_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(dev_out.cpu() - cpu_out)).item())
    print("Timesteps simulation done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet (diffusers) test with comparison")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Diffusers model id (e.g., Qwen/Qwen-Image)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--prompt", type=str, default="a cat in a hat")
    args = parser.parse_args()

    sys.path.append(os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
    device = torch.device("npu:0")

    #test_upsample2d(device)
    #test_groupnorm(device)
    #test_groupnorm(device, stride=[1, 1, 320*32, 320])
    #test_resnetblock2d(device, in_channels=640, out_channels=320, temb_channels=256, resnet_act_fn='silu')
    #test_resnetblock2d(device, in_channels=640, out_channels=320, temb_channels=1280)
    #test_cross_attn_down_block2d(device)
    #test_unet_mid_block2d_cross_attn(device)
    #test_cross_attn_up_block2d(device)
    #test_flip_sin_to_cos_embedding(device)
    #test_timesteps(device)
    test_unet2d_condition_model(device)
    #test_unet_conditional(
    #    device=device,
    #    model_id=args.model,
    #    batch=args.batch,
    #    dtype=args.dtype,
    #    rtol=args.rtol,
    #    atol=args.atol,
    #    prompt=args.prompt,
    #)