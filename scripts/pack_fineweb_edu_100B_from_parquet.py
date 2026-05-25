import argparse
import importlib.util
import json
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


def write_manifest(path: Path, payload: dict) -> None:
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

    indices = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.bin$")
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


def prepare_builder(outdir: Path, prefix: str, chunk_size: int, eos_id: int, vocab_size: int, resume: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    count = existing_chunk_count(outdir, prefix, chunk_size)
    if count and not resume:
        raise FileExistsError(f"{outdir} already has {count} packed shard(s). Use --resume or clean it.")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack downloaded FineWeb-Edu parquet files into LitGPT .bin shards.")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-tokens", type=parse_tokens, default=parse_tokens("100B"))
    parser.add_argument("--val-tokens", type=parse_tokens, default=parse_tokens("50M"))
    parser.add_argument("--train-dir-name", default="train_100B")
    parser.add_argument("--val-dir-name", default="val_50M")
    parser.add_argument("--manifest-name", default="fineweb_edu_100B_manifest.json")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-block-size", type=int, default=4097)
    parser.add_argument("--val-block-size", type=int, default=16385)
    parser.add_argument("--train-chunk-blocks", type=int, default=4096)
    parser.add_argument("--val-chunk-blocks", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    parquet_files = sorted(args.parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.parquet_dir}")

    train_dir = args.out_root / args.train_dir_name
    val_dir = args.out_root / args.val_dir_name
    manifest_path = args.out_root / args.manifest_name
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(f"Dataset already complete according to {manifest_path}")
            return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {args.tokenizer!r} does not define eos_token_id")
    vocab_size = len(tokenizer)

    train_builder, committed_train_tokens = prepare_builder(
        train_dir, "train_slim", args.train_block_size * args.train_chunk_blocks, eos_id, vocab_size, args.resume
    )
    val_builder, committed_val_tokens = prepare_builder(
        val_dir, "validation", args.val_block_size * args.val_chunk_blocks, eos_id, vocab_size, args.resume
    )

    start = time.time()
    docs = 0
    train_tokens = committed_train_tokens
    val_tokens = committed_val_tokens
    skip_val_tokens = committed_val_tokens
    skip_train_tokens = committed_train_tokens
    current_file = None

    def checkpoint(status: str) -> None:
        elapsed = max(time.time() - start, 1e-6)
        total_new = train_tokens + val_tokens - committed_train_tokens - committed_val_tokens
        write_manifest(
            manifest_path,
            {
                "status": status,
                "source": "local_parquet",
                "parquet_dir": str(args.parquet_dir),
                "current_file": str(current_file) if current_file else None,
                "num_parquet_files": len(parquet_files),
                "docs_seen_this_run": docs,
                "val_tokens": val_tokens,
                "train_tokens": train_tokens,
                "committed_val_tokens_at_start": committed_val_tokens,
                "committed_train_tokens_at_start": committed_train_tokens,
                "target_val_tokens": args.val_tokens,
                "target_train_tokens": args.train_tokens,
                "elapsed_seconds_this_run": elapsed,
                "tokens_per_second_this_run": total_new / elapsed,
                "train_dir": str(train_dir),
                "val_dir": str(val_dir),
                "resume_safe_note": "Resume uses existing complete shards as the checkpoint. Tokens in an unwritten partial chunk may be recomputed.",
            },
        )

    checkpoint("running")
    for parquet_path in parquet_files:
        current_file = parquet_path
        if train_tokens >= args.train_tokens and val_tokens >= args.val_tokens:
            break
        print(f"Reading {parquet_path}", flush=True)
        reader = pq.ParquetFile(parquet_path)
        for batch in reader.iter_batches(batch_size=args.batch_size, columns=[args.text_column]):
            if train_tokens >= args.train_tokens and val_tokens >= args.val_tokens:
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
                    n = add_tokens(val_builder, ids[:remaining])
                    val_tokens += n
                    ids = ids[n:]
                if skip_train_tokens:
                    n = min(len(ids), skip_train_tokens)
                    skip_train_tokens -= n
                    ids = ids[n:]
                if ids and train_tokens < args.train_tokens:
                    remaining = args.train_tokens - train_tokens
                    train_tokens += add_tokens(train_builder, ids[:remaining])
                if docs % 10000 == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    total_new = train_tokens + val_tokens - committed_train_tokens - committed_val_tokens
                    print(
                        f"docs={docs:,} val={val_tokens:,}/{args.val_tokens:,} "
                        f"train={train_tokens:,}/{args.train_tokens:,} "
                        f"new_rate={total_new / elapsed:,.0f} tok/s",
                        flush=True,
                    )
                    checkpoint("running")
        checkpoint("running")

    if val_tokens:
        val_builder.write_reminder()
    if train_tokens:
        train_builder.write_reminder()
    checkpoint("complete" if train_tokens >= args.train_tokens and val_tokens >= args.val_tokens else "incomplete")


if __name__ == "__main__":
    main()
