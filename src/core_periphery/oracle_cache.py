
import os
import json
import glob
import hashlib
import time
import uuid
import numpy as np

from oracle import compute_oracle_table

DEFAULT_CACHE_DIR = "oracle_cache"
_ROUND_DECIMALS = 10


def _cache_key(mu1_star, mu2_star, v, p, T):
    canonical = (
        round(float(mu1_star), _ROUND_DECIMALS),
        round(float(mu2_star), _ROUND_DECIMALS),
        round(float(v), _ROUND_DECIMALS),
        round(float(p), _ROUND_DECIMALS),
        int(T),
    )
    s = repr(canonical)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def get_oracle_table(mu1_star, mu2_star, v, p, T, cache_dir=DEFAULT_CACHE_DIR,
                      force_rebuild=False, verbose=True):

    key = _cache_key(mu1_star, mu2_star, v, p, T)
    filename = f"oracle_{key}.npy"
    filepath = os.path.join(cache_dir, filename)

    if not force_rebuild and os.path.exists(filepath):
        if verbose:
            print(f"[oracle_cache] HIT  key={key}")
        return np.load(filepath)

    if verbose:
        print(f"[oracle_cache] MISS key={key}  building...")
    t0 = time.time()
    table = compute_oracle_table(mu1_star, mu2_star, v, p, T)
    build_time = time.time() - t0

    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = os.path.join(cache_dir, f".tmp_{key}_{uuid.uuid4().hex}.npy")
    np.save(tmp_path, table)
    os.replace(tmp_path, filepath)  # atomic on POSIX; last writer wins, both are identical anyway

    if verbose:
        print(f"[oracle_cache] saved {filename}  ({build_time:.3f}s)")
    return table


def rebuild_manifest(cache_dir=DEFAULT_CACHE_DIR):
    manifest = {}
    for filepath in glob.glob(os.path.join(cache_dir, "oracle_*.npy")):
        filename = os.path.basename(filepath)
        key = filename[len("oracle_"):-len(".npy")]
        manifest[key] = {
            "filename": filename,
            "size_bytes": os.path.getsize(filepath),
        }
    manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[oracle_cache] rebuilt manifest.json: {len(manifest)} entries")
    return manifest


def cache_stats(cache_dir=DEFAULT_CACHE_DIR):
    files = glob.glob(os.path.join(cache_dir, "oracle_*.npy"))
    total_bytes = sum(os.path.getsize(f) for f in files)
    print(f"{len(files)} cached oracle tables, {total_bytes/1e6:.1f} MB total, in {cache_dir}")
    return files