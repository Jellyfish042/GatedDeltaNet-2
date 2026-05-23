########################################################################################################
# RWKV-7 CUDA Kernel Wrapper
# From https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HEAD_SIZE = 64  # default head size


def load_wkv7_cuda_kernel(head_size=64, use_training_kernel=False):
    global HEAD_SIZE
    HEAD_SIZE = head_size

    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_dir = os.path.join(cuda_dir, "cuda")

    if use_training_kernel:
        CHUNK_LEN = 16
        flags = [
            "-res-usage",
            f"-D_C_={head_size}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas",
            "-O3",
            "--extra-device-vectorization",
        ]
        load(
            name="wind_backstepping",
            sources=[os.path.join(cuda_dir, "wkv7_cuda.cu"), os.path.join(cuda_dir, "wkv7_cuda_op.cpp")],
            is_python_module=False,
            verbose=True,
            extra_cuda_cflags=flags,
        )
    else:
        flags = [
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas",
            "-O3",
            "--extra-device-vectorization",
            f"-D_N_={head_size}",
        ]
        load(
            name="wkv7",
            sources=[os.path.join(cuda_dir, "wkv7_op.cpp"), os.path.join(cuda_dir, "wkv7.cu")],
            is_python_module=False,
            verbose=True,
            extra_cuda_cflags=flags,
        )


class WindBackstepping(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        CHUNK_LEN = 16
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])

        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)

        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.dtype == torch.bfloat16
        assert dy.is_contiguous()

        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]

        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    B, T, HC = q.shape
    CHUNK_LEN = 16

    if T % CHUNK_LEN != 0:
        pad_len = CHUNK_LEN - (T % CHUNK_LEN)
        q = F.pad(q, (0, 0, 0, pad_len))
        w = F.pad(w, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        a = F.pad(a, (0, 0, 0, pad_len))
        b = F.pad(b, (0, 0, 0, pad_len))
    else:
        pad_len = 0

    T_padded = q.shape[1]
    q, w, k, v, a, b = [i.view(B, T_padded, HC // 64, 64) for i in [q, w, k, v, a, b]]
    out = WindBackstepping.apply(w, q, k, v, a, b).view(B, T_padded, HC)

    if pad_len > 0:
        out = out[:, :T, :].contiguous()

    return out


class WKV_7(torch.autograd.Function):

    @staticmethod
    def forward(ctx, r, w, k, v, a, b, dtype):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert r.is_contiguous()
            assert w.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert a.is_contiguous()
            assert b.is_contiguous()

            y = torch.empty((B, T, C), device=k.device, dtype=dtype, memory_format=torch.contiguous_format)
            torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y


def RWKV7_OP(r, w, k, v, a, b, dtype=torch.half):
    return WKV_7.apply(r, w, k, v, a, b, dtype)
