import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import re
import struct
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def load_packed_dataset_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "lit_gpt" / "packed_dataset.py"
    spec = importlib.util.spec_from_file_location("gdn2_packed_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


packed_dataset = load_packed_dataset_module()
PackedDatasetBuilder = packed_dataset.PackedDatasetBuilder


def parse_tokens(value: str) -> int:
    text = value.strip().lower().replace("_", "")
    multipliers = {"k": 10**3, "m": 10**6, "b": 10**9, "t": 10**12}
    if text[-1:] in multipliers:
        return int(float(text[:-1]) * multipliers[text[-1]])
    return int(text)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def add_tokens(builder: PackedDatasetBuilder, token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    arr = np.asarray(token_ids, dtype=builder.dtype)
    builder.add_array(arr)
    return len(token_ids)


def existing_chunk_count(outdir: Path, prefix: str, chunk_size: int) -> int:
    files = sorted(outdir.glob(f"{prefix}_*.bin"))
    if not files:
        return 0

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.bin$")
    indices = []
    for path in files:
        match = pattern.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected packed shard name: {path}")
        indices.append(int(match.group(1)))
        with path.open("rb") as f:
            magic = f.read(len(packed_dataset.HDR_MAGIC))
            if magic != packed_dataset.HDR_MAGIC:
                raise ValueError(f"Bad packed shard magic: {path}")
            version = struct.unpack("<Q", f.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported packed shard version {version}: {path}")
            f.read(1)
            file_chunk_size = struct.unpack("<Q", f.read(8))[0]
            if file_chunk_size != chunk_size:
                raise ValueError(f"Chunk size mismatch in {path}: {file_chunk_size} != {chunk_size}")
    if indices != list(range(len(indices))):
        raise ValueError(f"Packed shards for {prefix!r} are not contiguous from 0")
    return len(indices)


def make_builder(outdir: Path, prefix: str, chunk_size: int, eos_id: int, vocab_size: int, resume: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    count = existing_chunk_count(outdir, prefix, chunk_size)
    if count and not resume:
        raise FileExistsError(f"{outdir} already has shards for {prefix}. Use --resume or clean it.")
    builder = PackedDatasetBuilder(
        outdir=outdir,
        prefix=prefix,
        chunk_size=chunk_size,
        sep_token=eos_id,
        dtype="auto",
        vocab_size=vocab_size,
    )
    builder._counter = count
    return builder, count * chunk_size


def worker_main(args_dict: dict) -> None:
    worker_id = args_dict["worker_id"]
    parquet_files = [Path(p) for p in args_dict["parquet_files"]]
    out_root = Path(args_dict["out_root"])
    train_dir = out_root / args_dict["train_dir_name"]
    tokenizer_path = args_dict["tokenizer"]
    batch_size = args_dict["batch_size"]
    text_column = args_dict["text_column"]
    train_chunk_size = args_dict["train_block_size"] * args_dict["train_chunk_blocks"]
    resume = args_dict["resume"]
    manifest_dir = out_root / "parallel_pack_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"worker_{worker_id:03d}.json"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {tokenizer_path!r} does not define eos_token_id")
    builder, committed_tokens = make_builder(
        train_dir,
        f"train_slim_w{worker_id:03d}",
        train_chunk_size,
        eos_id,
        len(tokenizer),
        resume,
    )

    start = time.time()
    docs = 0
    tokens = committed_tokens
    skip_tokens = committed_tokens
    current_file = None

    def checkpoint(status: str) -> None:
        elapsed = max(time.time() - start, 1e-6)
        write_json(
            manifest_path,
            {
                "status": status,
                "worker_id": worker_id,
                "files_assigned": len(parquet_files),
                "current_file": str(current_file) if current_file else None,
                "docs_seen_this_run": docs,
                "committed_tokens_at_start": committed_tokens,
                "train_tokens": tokens,
                "new_tokens_this_run": tokens - committed_tokens,
                "elapsed_seconds_this_run": elapsed,
                "tokens_per_second_this_run": (tokens - committed_tokens) / elapsed,
                "prefix": f"train_slim_w{worker_id:03d}",
            },
        )

    checkpoint("running")
    for parquet_path in parquet_files:
        current_file = parquet_path
        print(f"[worker {worker_id}] Reading {parquet_path}", flush=True)
        reader = pq.ParquetFile(parquet_path)
        for batch in reader.iter_batches(batch_size=batch_size, columns=[text_column]):
            encoded = tokenizer(batch.column(0).to_pylist(), add_special_tokens=False)["input_ids"]
            for ids in encoded:
                docs += 1
                if not ids:
                    continue
                ids = ids + [eos_id]
                if skip_tokens:
                    n = min(len(ids), skip_tokens)
                    skip_tokens -= n
                    ids = ids[n:]
                if ids:
                    tokens += add_tokens(builder, ids)
                if docs % 10000 == 0:
                    checkpoint("running")
        checkpoint("running")

    if tokens:
        builder.write_reminder()
    checkpoint("complete")


def pack_validation(args, parquet_files: list[Path], eos_id: int, vocab_size: int, tokenizer) -> None:
    val_dir = args.out_root / args.val_dir_name
    val_chunk_size = args.val_block_size * args.val_chunk_blocks
    val_builder, committed_val_tokens = make_builder(
        val_dir,
        "validation",
        val_chunk_size,
        eos_id,
        vocab_size,
        args.resume,
    )
    if committed_val_tokens >= args.val_tokens:
        return

    start = time.time()
    docs = 0
    val_tokens = committed_val_tokens
    skip_val_tokens = committed_val_tokens
    current_file = None
    manifest_path = args.out_root / "fineweb_edu_100B_val_pack_manifest.json"

    def checkpoint(status: str) -> None:
        elapsed = max(time.time() - start, 1e-6)
        write_json(
            manifest_path,
            {
                "status": status,
                "source": "local_parquet_validation",
                "current_file": str(current_file) if current_file else None,
                "docs_seen_this_run": docs,
                "committed_val_tokens_at_start": committed_val_tokens,
                "val_tokens": val_tokens,
                "target_val_tokens": args.val_tokens,
                "elapsed_seconds_this_run": elapsed,
                "tokens_per_second_this_run": (val_tokens - committed_val_tokens) / elapsed,
                "val_dir": str(val_dir),
            },
        )

    checkpoint("running")
    for parquet_path in parquet_files:
        current_file = parquet_path
        if val_tokens >= args.val_tokens:
            break
        print(f"[validation] Reading {parquet_path}", flush=True)
        reader = pq.ParquetFile(parquet_path)
        for batch in reader.iter_batches(batch_size=args.batch_size, columns=[args.text_column]):
            if val_tokens >= args.val_tokens:
                break
            encoded = tokenizer(batch.column(0).to_pylist(), add_special_tokens=False)["input_ids"]
            for ids in encoded:
                docs += 1
                if not ids:
                    continue
                ids = ids + [eos_id]
                if skip_val_tokens:
                    n = min(len(ids), skip_val_tokens)
                    skip_val_tokens -= n
                    ids = ids[n:]
                if ids and val_tokens < args.val_tokens:
                    remaining = args.val_tokens - val_tokens
                    val_tokens += add_tokens(val_builder, ids[:remaining])
                if docs % 10000 == 0:
                    checkpoint("running")
        checkpoint("running")

    if val_tokens:
        val_builder.write_reminder()
    checkpoint("complete" if val_tokens >= args.val_tokens else "incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel-pack downloaded FineWeb-Edu parquet files into LitGPT .bin shards.")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--train-dir-name", default="train_100B")
    parser.add_argument("--val-dir-name", default="val_50M")
    parser.add_argument("--val-tokens", type=parse_tokens, default=parse_tokens("50M"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--train-block-size", type=int, default=4097)
    parser.add_argument("--train-chunk-blocks", type=int, default=4096)
    parser.add_argument("--val-block-size", type=int, default=16385)
    parser.add_argument("--val-chunk-blocks", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    parquet_files = sorted(args.parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.parquet_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {args.tokenizer!r} does not define eos_token_id")
    pack_validation(args, parquet_files, eos_id, len(tokenizer), tokenizer)

    workers = max(1, min(args.workers, len(parquet_files)))
    shards = [parquet_files[i::workers] for i in range(workers)]
    payloads = []
    for worker_id, files in enumerate(shards):
        payloads.append(
            {
                "worker_id": worker_id,
                "parquet_files": [str(path) for path in files],
                "out_root": str(args.out_root),
                "tokenizer": args.tokenizer,
                "train_dir_name": args.train_dir_name,
                "text_column": args.text_column,
                "batch_size": args.batch_size,
                "train_block_size": args.train_block_size,
                "train_chunk_blocks": args.train_chunk_blocks,
                "resume": args.resume,
            },
        )

    write_json(
        args.out_root / "fineweb_edu_100B_parallel_pack_manifest.json",
        {
            "status": "running",
            "source": "local_parquet_parallel",
            "parquet_dir": str(args.parquet_dir),
            "num_parquet_files": len(parquet_files),
            "workers": workers,
            "note": "Workers write train_slim_wNNN_* shards. pretrain.py glob train_slim* reads them directly.",
        },
    )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(worker_main, payloads)

    manifest_dir = args.out_root / "parallel_pack_manifests"
    worker_manifests = sorted(manifest_dir.glob("worker_*.json"))
    total_tokens = 0
    completed = 0
    for path in worker_manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        total_tokens += int(payload.get("train_tokens", 0))
        completed += payload.get("status") == "complete"

    write_json(
        args.out_root / "fineweb_edu_100B_parallel_pack_manifest.json",
        {
            "status": "complete" if completed == len(worker_manifests) else "incomplete",
            "source": "local_parquet_parallel",
            "parquet_dir": str(args.parquet_dir),
            "num_parquet_files": len(parquet_files),
            "workers": workers,
            "worker_manifests": len(worker_manifests),
            "total_train_tokens": total_tokens,
            "train_dir": str(args.out_root / args.train_dir_name),
        },
    )


if __name__ == "__main__":
    main()
