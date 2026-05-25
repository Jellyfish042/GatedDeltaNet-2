import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu sample-100BT parquet shards with HF Hub.")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--allow-pattern", default="sample/100BT/*.parquet")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--manifest-name", default="download_manifest.json")
    args = parser.parse_args()

    start = time.time()
    manifest_path = args.local_dir / args.manifest_name
    write_json(
        manifest_path,
        {
            "status": "running",
            "repo_id": args.repo_id,
            "revision": args.revision,
            "allow_pattern": args.allow_pattern,
            "local_dir": str(args.local_dir),
            "max_workers": args.max_workers,
        },
    )
    resolved = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=args.allow_pattern,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
        resume_download=True,
    )
    parquet_files = sorted(args.local_dir.rglob("*.parquet"))
    total_bytes = sum(path.stat().st_size for path in parquet_files)
    elapsed = time.time() - start
    write_json(
        manifest_path,
        {
            "status": "complete",
            "repo_id": args.repo_id,
            "revision": args.revision,
            "allow_pattern": args.allow_pattern,
            "local_dir": str(args.local_dir),
            "resolved_dir": resolved,
            "max_workers": args.max_workers,
            "parquet_files": len(parquet_files),
            "total_bytes": total_bytes,
            "elapsed_seconds": elapsed,
            "bytes_per_second": total_bytes / max(elapsed, 1e-6),
        },
    )
    print(f"Downloaded {len(parquet_files)} parquet files, {total_bytes / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
