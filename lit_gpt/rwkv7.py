from types import SimpleNamespace

import torch
import torch.nn as nn

from lit_gpt.rwkv7_ops.rwkv7_cuda_kernel import load_wkv7_cuda_kernel
from lit_gpt.rwkv7_ops.time_mix import RWKV_Tmix_x070

_LOADED_HEAD_SIZES: set[int] = set()


def _ensure_rwkv7_kernel(head_size: int) -> None:
    if head_size in _LOADED_HEAD_SIZES:
        return
    load_wkv7_cuda_kernel(head_size=head_size, use_training_kernel=True)
    _LOADED_HEAD_SIZES.add(head_size)


class RWKV7TimeMix(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_idx: int,
        head_size: int = 64,
        head_size_divisor: int = 8,
    ) -> None:
        super().__init__()
        if n_embd % head_size != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by rwkv7 head_size={head_size}")
        _ensure_rwkv7_kernel(head_size)
        args = SimpleNamespace(
            n_embd=n_embd,
            dim_att=n_embd,
            n_layer=n_layer,
            head_size_a=head_size,
            head_size_divisor=head_size_divisor,
            my_testing="x070",
        )
        self.time_mix = RWKV_Tmix_x070(args, layer_idx)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if v_first is None:
            v_first = torch.empty_like(x)
        return self.time_mix(x, v_first)
