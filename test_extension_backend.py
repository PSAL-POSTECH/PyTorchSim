import torch._dynamo
import torch.utils.cpp_extension
from tests.test_add import test_vectoradd
from tests.test_reduce import test_reduce_sum
from tests.test_transpose2D import test_Transpose2D, test_Transpose2D_2
from tests.test_transpose3D import test_Transpose3D_1, test_Transpose3D_2, test_Transpose3D_3
from tests.test_view3D_2D import test_view3D_2D
from tests.test_softmax import test_softmax
from tests.test_batchnorm import test_BatchNorm
from tests.test_layernorm import test_LayerNorm
from tests.test_conv2d import test_conv2d
from tests.test_matmul import test_matmul
from tests.test_bmm import test_BMM
from tests.test_cnn import test_CNN
from tests.test_transformer import test_DecoderBlock
from tests.test_resnet import test_resnet
from tests.test_mlp import test_mlp, test_mlp_inf
from tests.MoE.test_moe import test_moe
from tests.test_pool import test_avgpool, test_maxpool
from tests.Fusion.test_addmm_residual import test_addmm_residual
from tests.Fusion.test_matmul_scalar import test_matmul_scalar
from tests.Fusion.test_matmul_activation import test_matmul_activation

if __name__ == "__main__":
    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_vectoradd(device, (47, 10))
    test_reduce_sum(device, (29, 47), 1, keepdim=True)
    test_reduce_sum(device, (17, 68), 0, keepdim=True)
    test_Transpose2D(device, [64, 156])
    test_Transpose2D_2(device, [16, 64])
    test_Transpose3D_1(device, [62, 34, 44])
    test_Transpose3D_2(device, [62, 34, 44])
    test_Transpose3D_3(device, [62, 34, 44])
    test_view3D_2D(device)
    test_maxpool(device)
    test_avgpool(device)
    test_softmax(device, (64, 128), dim=1)
    test_BatchNorm(device)
    test_LayerNorm(device, (64, 128))
    test_conv2d(device)
    test_matmul(device, 33, 45, 68)
    test_BMM(device)
    test_CNN(device)
    test_DecoderBlock(device)
    test_resnet(device)
    test_mlp(device)
    test_mlp_inf(device, batch_size=64, input_size=256, hidden_size=512, output_size=256, sparsity=0.97)

    # # Fusion Test
    test_matmul_scalar(device)
    test_matmul_activation(device, batch_size=32, input_size=32, output_size=32, activation_fn="relu")
    test_matmul_activation(device, batch_size=32, input_size=32, output_size=32, activation_fn="sigmoid")
    test_addmm_residual(device)
