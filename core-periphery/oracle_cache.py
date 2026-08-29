"""
Disk cache for oracle tables, keyed by (mu1_star, mu2_star, v, p, T).

Design:
  - Each unique parameter combination's action_table is saved as its own
    .npy file (binary, compact, fast) -- NOT JSON, since a (T+1)x(T+1)
    int8 array (~25MB at T=5000) would balloon to 100+MB as JSON text
    and be slow to parse.
  - The .npy FILENAME ITSELF (derived deterministically from a hash of
    the parameters) is the source of truth for hit/miss -- existence is
    checked directly via os.path.exists, with no shared index required
    for correctness. This matters because this cache may be hit by many
    processes/containers concurrently (e.g. a Modal sweep dispatching
    ~100 parallel containers against a shared Volume); a manifest that
    required read-modify-write on every access would race and corrupt
    under that load.
  - manifest.json is kept as an OPTIONAL, regenerable convenience index
    for human inspection (via rebuild_manifest()) -- it is never
    required to be read or updated on the hot path, so it can't corrupt
    under concurrent access.

Usage:
    from oracle_cache import get_oracle_table
    table = get_oracle_table(mu1_star=0.6, mu2_star=0.75, v=0.01, p=0.85, T=5000)
    # optionally point at a different cache directory (e.g. a mounted
    # Modal Volume path) via cache_dir=...
"""

import os
import json
import glob
import hashlib
import time
import uuid
import numpy as np

from core_periphery_bandit import compute_oracle_table

DEFAULT_CACHE_DIR = "oracle_cache"

# Round floats to this many decimals before hashing, so that
# floating-point representation noise (e.g. from np.linspace) doesn't
# cause two "same" parameter values to miss each other in the cache.
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
    """Returns the oracle action_table for these parameters. Builds and
    caches to disk on first call (from ANY process); loads from disk on
    every subsequent call with the same (mu1_star, mu2_star, v, p, T),
    including from a completely different process/container, as long as
    they share the same cache_dir (e.g. a mounted Modal Volume).

    Safe under concurrent callers: hit/miss is decided purely by file
    existence, and writes go to a unique temp file then get atomically
    renamed into place, so two processes racing on the same cache MISS
    can both build and write without corrupting each other (worst case:
    duplicated work that round, never a corrupted or partially-written
    cache file)."""
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
    # write to a unique temp file, then atomically rename -- safe even if
    # another process is concurrently building the SAME key right now
    tmp_path = os.path.join(cache_dir, f".tmp_{key}_{uuid.uuid4().hex}.npy")
    np.save(tmp_path, table)
    os.replace(tmp_path, filepath)  # atomic on POSIX; last writer wins, both are identical anyway

    if verbose:
        print(f"[oracle_cache] saved {filename}  ({build_time:.3f}s)")
    return table


def rebuild_manifest(cache_dir=DEFAULT_CACHE_DIR):
    """Scans the cache directory and regenerates manifest.json from
    whatever .npy files actually exist. Safe to call any time (e.g.
    once, after a parallel sweep finishes) since it only READS existing
    files -- never called on the hot path, so no concurrency concern."""
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