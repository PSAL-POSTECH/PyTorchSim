"""one_layer._drop_plain_mask: a decode's all-allowing mask and a prefill's plain causal one (bool or
additive) are dropped, as the TPU rows pass none, so transformers hands SDPA the shared KV heads rather
than repeat_kv's copies; a sliding window or a padded key keeps its mask."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from one_layer import _drop_plain_mask  # noqa: E402

CAUSAL = torch.ones(128, 128, dtype=torch.bool).tril()[None, None]


def _dropped(mask):
    return _drop_plain_mask({"attention_mask": mask, "use_cache": True})["attention_mask"] is None


def test_plain_masks_are_dropped():
    assert _dropped(torch.ones(1, 1, 1, 1024, dtype=torch.bool))
    assert _dropped(CAUSAL)
    assert _dropped(torch.zeros(1, 1, 128, 128).masked_fill(~CAUSAL, torch.finfo(torch.float32).min))


def test_masks_that_say_more_are_kept():
    window = CAUSAL & torch.ones(128, 128, dtype=torch.bool).triu(-31)
    padded = torch.ones(1, 1, 1, 1024, dtype=torch.bool)
    padded[..., :5] = False
    assert not _dropped(window) and not _dropped(padded)
    assert not _dropped(torch.zeros(1, 1, 128, 128).masked_fill(~CAUSAL, -1.0))


if __name__ == "__main__":
    test_plain_masks_are_dropped()
    test_masks_that_say_more_are_kept()
    print("PASS")
