import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


CHUNK_LEN = 16
HEAD_SIZE = 64
_LOADED = False
_REGISTERED = False


def _cuda_file(name: str) -> str:
    return str(Path(__file__).resolve().parent / "cuda" / name)


def _load_extension(name: str, sources: list[str], extra_cuda_cflags: list[str] | None = None) -> None:
    verbose = os.getenv("RWKV7_KERNEL_VERBOSE", "0") == "1"
    load(
        name=name,
        sources=[_cuda_file(source) for source in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags
        or ["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
        is_python_module=False,
        verbose=verbose,
    )


def ensure_fused_time_mix_kernels(head_size: int = HEAD_SIZE) -> None:
    global _LOADED, _REGISTERED
    if head_size != HEAD_SIZE:
        raise ValueError("The fused RWKV-7 TimeMix kernels currently require head_size=64")
    if _LOADED:
        return

    clampw_flags = [
        "-res-usage",
        f"-D_N_={head_size}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]
    _load_extension(
        "rwkv7_clampw_v3",
        ["rwkv7_clampw_v3_for_h100.cu", "rwkv7_clampw_v3.cpp"],
        extra_cuda_cflags=clampw_flags,
    )
    _load_extension("rwkv7_tmix_mix6_bf16_v5", ["rwkv7_tmix_mix6_bf16_v5.cpp", "rwkv7_tmix_mix6_bf16_v5.cu"])
    _load_extension("rwkv7_tmix_kk_pre_bf16_v5", ["rwkv7_tmix_kk_pre_bf16_v5.cpp", "rwkv7_tmix_kk_pre_bf16_v5.cu"])
    _load_extension(
        "rwkv7_tmix_lnx_rkvres_xg_bf16_v1",
        ["rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cpp", "rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu"],
    )
    _load_extension("rwkv7_tmix_a_gate_bf16", ["rwkv7_tmix_a_gate_bf16.cpp", "rwkv7_tmix_a_gate_bf16.cu"])
    _load_extension(
        "rwkv7_tmix_vres_gate_bf16_v1",
        ["rwkv7_tmix_vres_gate_bf16_v1.cpp", "rwkv7_tmix_vres_gate_bf16_v1.cu"],
    )

    if not _REGISTERED:
        _register_autograd()
        _REGISTERED = True
    _LOADED = True


def _bf16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16 and x.is_contiguous():
        return x
    return x.contiguous().to(torch.bfloat16)


class _RWKV7ClampWFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        B, T, H, N = r.shape
        if T % CHUNK_LEN != 0:
            raise ValueError(f"RWKV-7 clampw kernel requires T divisible by {CHUNK_LEN}; got T={T}")
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)
        torch.ops.rwkv7_clampw_v3.forward(r, w, k, v, a, b, y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in (r, w, k, v, a, b)]
        torch.ops.rwkv7_clampw_v3.backward(r, w, k, v, a, b, dy.contiguous(), s, sa, dr, dw, dk, dv, da, db)
        return dr, dw, dk, dv, da, db


def rwkv7_clampw_cuda(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, T, C = r.shape
    pad_len = (CHUNK_LEN - T % CHUNK_LEN) % CHUNK_LEN
    tensors = [_bf16(x) for x in (r, w, k, v, a, b)]
    if pad_len:
        tensors = [F.pad(x, (0, 0, 0, pad_len)) for x in tensors]
    Tp = tensors[0].shape[1]
    tensors = [x.view(B, Tp, C // HEAD_SIZE, HEAD_SIZE).contiguous() for x in tensors]
    out = _RWKV7ClampWFn.apply(*tensors).view(B, Tp, C)
    if pad_len:
        out = out[:, :T, :].contiguous()
    return out


def _register_autograd() -> None:
    def mix6_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs)

    def mix6_backward(ctx, grads):
        return tuple(
            torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(
                grads[0].contiguous(),
                grads[1].contiguous(),
                grads[2].contiguous(),
                grads[3].contiguous(),
                grads[4].contiguous(),
                grads[5].contiguous(),
                *ctx.saved_tensors,
            )
        )

    torch.library.register_autograd("rwkv7_tmix_mix6_bf16_v5::forward", mix6_backward, setup_context=mix6_setup)

    def kk_setup(ctx, inputs, output):
        k, k_k, a, k_a, _head_size = inputs
        del _head_size
        ctx.save_for_backward(k, k_k, a, k_a, output[3])

    def kk_backward(ctx, grads):
        k, k_k, a, k_a, inv_d = ctx.saved_tensors
        return tuple(
            torch.ops.rwkv7_tmix_kk_pre_bf16_v5.backward(
                grads[0].contiguous(),
                grads[1].contiguous(),
                grads[2].contiguous(),
                k,
                k_k,
                a,
                k_a,
                inv_d,
                HEAD_SIZE,
            )
        ) + (None,)

    torch.library.register_autograd("rwkv7_tmix_kk_pre_bf16_v5::forward", kk_backward, setup_context=kk_setup)

    def lnx_setup(ctx, inputs, output):
        x, r, k, v, r_k, weight, bias, g = inputs
        ctx.save_for_backward(x, r, k, v, r_k, weight, bias, g, output[1], output[2])

    def lnx_backward(ctx, grads):
        x, r, k, v, r_k, weight, bias, g, mean, rstd = ctx.saved_tensors
        return tuple(
            torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.backward(
                grads[0].contiguous(),
                x,
                r,
                k,
                v,
                r_k,
                weight,
                bias,
                g,
                mean,
                rstd,
            )
        )

    torch.library.register_autograd("rwkv7_tmix_lnx_rkvres_xg_bf16_v1::forward", lnx_backward, setup_context=lnx_setup)

    def a_gate_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs)

    def a_gate_backward(ctx, grad_out):
        a0, a12 = ctx.saved_tensors
        return tuple(torch.ops.rwkv7_tmix_a_gate_bf16.backward(grad_out.contiguous(), a0, a12))

    torch.library.register_autograd("rwkv7_tmix_a_gate_bf16::forward", a_gate_backward, setup_context=a_gate_setup)

    def vres_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs)

    def vres_backward(ctx, grad_out):
        v, v_first, v0, v12 = ctx.saved_tensors
        grad_v, grad_v_first, grad_pre = torch.ops.rwkv7_tmix_vres_gate_bf16_v1.backward(
            grad_out.contiguous(),
            v,
            v_first,
            v0,
            v12,
        )
        grad_v0 = grad_pre.sum(dim=(0, 1), keepdim=True)
        return grad_v, grad_v_first, grad_v0.to(v0.dtype), grad_pre.to(v12.dtype)

    torch.library.register_autograd("rwkv7_tmix_vres_gate_bf16_v1::forward", vres_backward, setup_context=vres_setup)


def tmix_mix6_bf16_v5(x, x_r, x_w, x_k, x_v, x_a, x_g):
    outs = torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(
        _bf16(x),
        _bf16(x_r.view(-1)),
        _bf16(x_w.view(-1)),
        _bf16(x_k.view(-1)),
        _bf16(x_v.view(-1)),
        _bf16(x_a.view(-1)),
        _bf16(x_g.view(-1)),
    )
    return outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]


def tmix_kk_pre_bf16_v5(k, k_k, a, k_a):
    outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
        _bf16(k),
        _bf16(k_k.view(-1)),
        _bf16(a),
        _bf16(k_a.view(-1)),
        HEAD_SIZE,
    )
    return outs[0], outs[1], outs[2]


def tmix_lnx_rkvres_xg_bf16_v1(x, r, k, v, r_k, weight, bias, g):
    outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(
        _bf16(x),
        _bf16(r),
        _bf16(k),
        _bf16(v),
        _bf16(r_k),
        _bf16(weight.view(-1)),
        _bf16(bias.view(-1)),
        _bf16(g),
    )
    return outs[0]


def tmix_a_gate_bf16(a0, a12):
    return torch.ops.rwkv7_tmix_a_gate_bf16.forward(_bf16(a0.view(-1)), _bf16(a12))


def tmix_vres_gate_bf16_v1(v, v_first, v0, v12):
    return torch.ops.rwkv7_tmix_vres_gate_bf16_v1.forward(_bf16(v), _bf16(v_first), _bf16(v0.view(-1)), _bf16(v12))
