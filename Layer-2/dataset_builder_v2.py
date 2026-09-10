

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LABEL_NAMES = {0: "ham", 1: "phishing", 2: "ai_phish"}
LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}


# ---------------------------------------------------------------------------
# Normalisation — strip the things campaign variants tweak
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def light_normalize(text: str) -> str:
    """Conservative normalisation used for EXACT-duplicate deletion only.

    Lowercase + collapse whitespace, nothing else. Two rows are treated as
    the same row only if they differ purely by case/spacing. We deliberately
    do NOT blank numbers or URLs here: two emails differing only by an
    invoice number are genuinely different samples and deleting one loses
    real signal. Template variance is handled by CLUSTERING (below) instead,
    which groups them without discarding any of them.
    """
    return _WS_RE.sub(" ", text.lower()).strip()


def aggressive_normalize(text: str) -> str:
    """Aggressive normalisation used ONLY to generate near-duplicate
    candidates for clustering — never to delete rows.

    Campaigns typically vary: recipient name, an amount, a deadline date, a
    tracking number, and the payload URL. Blanking those makes two variants
    of one template shingle almost identically, so they cluster into one
    group and therefore land in the same split.
    """
    t = text.lower()
    t = _URL_RE.sub(" <url> ", t)
    t = _EMAIL_RE.sub(" <email> ", t)
    t = _NUM_RE.sub(" <num> ", t)
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


# Kept for backwards compatibility with any caller importing the old name
normalize_text = aggressive_normalize


# ---------------------------------------------------------------------------
# Near-duplicate detection: shingles -> MinHash -> LSH banding
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 5) -> set[str]:
    """Character k-shingles. Character-level (not word) so that small word
    substitutions still overlap heavily."""
    t = aggressive_normalize(text)
    if len(t) < k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _signatures_matrix(texts: list[str], num_perm: int = 64, k: int = 5,
                       max_chars: int = 4000) -> "np.ndarray":
    """Vectorised MinHash signatures for every text at once.

    The v1 implementation computed an MD5 per (shingle, permutation) pair in
    pure Python: ~64 x 150 hashes per email, which projected to several
    minutes of signature building on 23.5k emails before the (worse)
    comparison stage even started.

    Here each shingle is hashed ONCE into a 64-bit integer, then all 64
    permutations are applied to that integer array in a single vectorised
    numpy op using the standard (a*x + b) mod p universal-hashing trick.
    That turns the inner loop into array arithmetic and gives a ~100x
    speedup while producing equivalent MinHash signatures.

    `max_chars` truncates very long emails — the leading few thousand
    characters are more than enough to identify a template, and it bounds
    worst-case cost on outlier-length messages.
    """
    import numpy as np

    rng = np.random.default_rng(12345)  # fixed seed -> reproducible signatures
    MERSENNE = (1 << 61) - 1
    a = rng.integers(1, MERSENNE, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=num_perm, dtype=np.uint64)

    sigs = np.full((len(texts), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)

    for i, text in enumerate(texts):
        sh = _shingles(text[:max_chars], k)
        if not sh:
            sigs[i, :] = 0
            continue
        # Hash each shingle ONCE (blake2b digest_size=8 -> fast 64-bit int)
        base = np.fromiter(
            (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
             for s in sh),
            dtype=np.uint64, count=len(sh),
        )
        # Apply all permutations at once: (a * x + b) mod p  -> (n_shingles, num_perm)
        perm = (np.outer(base, a) + b) % np.uint64(MERSENNE)
        sigs[i, :] = perm.min(axis=0)

    return sigs


def cluster_near_duplicates(texts: list[str], threshold: float = 0.85,
                            num_perm: int = 64, bands: int = 16,
                            verbose: bool = True) -> list[int]:
    """Return a cluster id per text; same cluster == near-duplicates.

    Two-stage, both stages kept cheap:
      1. Vectorised MinHash signatures (see _signatures_matrix).
      2. LSH banding to generate candidates, then union-find to merge.

    Crucially, within an LSH bucket we do NOT compare all pairs — that is
    what made large campaign buckets explode quadratically. Instead every
    member of a bucket is unioned to the bucket's first member. Items that
    agree on a whole band already have high estimated similarity, and
    union-find makes the merges transitive, so a campaign collapses to one
    cluster in linear time. Band width is derived from the threshold so the
    LSH sensitivity actually matches the requested similarity.
    """
    import numpy as np

    n = len(texts)
    if n == 0:
        return []

    if verbose:
        print(f"    computing MinHash signatures for {n} texts...")
    sigs = _signatures_matrix(texts, num_perm=num_perm)

    # Choose rows-per-band so the LSH S-curve turns on near `threshold`.
    # For b bands of r rows, P(candidate) = 1 - (1 - s^r)^b, whose steep
    # region sits near s ≈ (1/b)^(1/r). We pick the (b, r) split of num_perm
    # whose implied cutoff is closest to the requested threshold. The v1
    # heuristic (r ≈ 1/threshold²) gave r=3 at 0.6, which unioned documents
    # sharing any 3 hash values and collapsed the whole corpus into one
    # cluster — far too permissive.
    best = None
    for r in range(1, num_perm + 1):
        if num_perm % r:
            continue
        b = num_perm // r
        cutoff = (1.0 / b) ** (1.0 / r)
        score = abs(cutoff - threshold)
        if best is None or score < best[0]:
            best = (score, r, b)
    _, rows, bands = best
    if verbose:
        implied = (1.0 / bands) ** (1.0 / rows)
        print(f"    LSH: {bands} bands x {rows} rows "
              f"(implied cutoff ~{implied:.2f}, requested {threshold})")

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    # LSH proposes candidates; the signature comparison DISPOSES of them.
    # Skipping this verification is tempting for speed but catastrophic: a
    # single spurious band collision gets chained transitively by union-find,
    # and with many bands the entire corpus collapses into one component.
    # (Measured: 205 spurious band collisions across 2,000 genuinely distinct
    # documents were enough to merge all of them.) Verification is vectorised,
    # so it costs little.
    candidate_pairs = set()
    for band in range(bands):
        chunk = sigs[:, band * rows:(band + 1) * rows]
        keys = np.ascontiguousarray(chunk).view(
            np.dtype((np.void, chunk.dtype.itemsize * chunk.shape[1]))
        ).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        # Walk runs of identical band-keys; pair each member with the run's head
        start = 0
        for pos in range(1, len(sorted_keys) + 1):
            if pos == len(sorted_keys) or sorted_keys[pos] != sorted_keys[start]:
                if pos - start > 1:
                    members = order[start:pos]
                    head = int(members[0])
                    # Cap pairs generated by any single pathological bucket
                    for other in members[1:2001]:
                        candidate_pairs.add((head, int(other)))
                start = pos

    if verbose:
        print(f"    verifying {len(candidate_pairs)} LSH candidate pairs...")

    for x, y in candidate_pairs:
        if find(x) == find(y):
            continue
        est = float((sigs[x] == sigs[y]).mean())
        if est >= threshold:
            union(x, y)

    return [find(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Group-aware splitting
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42


def group_aware_split(df: pd.DataFrame, cfg: SplitConfig):
    """Split by GROUP, never by row.

    Every near-duplicate cluster is one indivisible group. Groups are shuffled
    and assigned greedily to the split that is furthest below its target size,
    which keeps the splits close to the requested proportions without ever
    breaking a group apart. Class balance is approximated (not forced) because
    forcing exact stratification would require splitting groups — the very
    thing that causes leakage.
    """
    import random
    rng = random.Random(cfg.seed)

    groups = df.groupby("group_id").indices  # group_id -> row positions
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    n_total = len(df)
    targets = {
        "train": n_total * (1 - cfg.test_size - cfg.val_size),
        "val": n_total * cfg.val_size,
        "test": n_total * cfg.test_size,
    }
    assigned = {"train": [], "val": [], "test": []}
    sizes = {"train": 0, "val": 0, "test": 0}

    # Largest groups first so big campaigns don't overshoot a small split
    group_ids.sort(key=lambda g: -len(groups[g]))
    for gid in group_ids:
        rows = groups[gid]
        # Assign to whichever split is proportionally most under target
        deficits = {k: (targets[k] - sizes[k]) / max(targets[k], 1) for k in targets}
        pick = max(deficits, key=deficits.get)
        assigned[pick].extend(rows)
        sizes[pick] += len(rows)

    out = {}
    for split, rows in assigned.items():
        out[split] = df.iloc[sorted(rows)].reset_index(drop=True)
    return out["train"], out["val"], out["test"]


# ---------------------------------------------------------------------------
# Loading + reporting
# ---------------------------------------------------------------------------

def load_csvs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{p} needs 'text' and 'label' columns; got {list(df.columns)}")
        if df["label"].dtype == object:
            df["label"] = df["label"].map(lambda x: LABEL_IDS.get(str(x).lower().strip(), x))
        if "source" not in df.columns:
            df["source"] = Path(p).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_source_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for src, g in df.groupby("source"):
        counts = g["label"].value_counts().sort_index().to_dict()
        stats[str(src)] = {
            "total": int(len(g)),
            "by_class": {LABEL_NAMES.get(int(k), str(k)): int(v) for k, v in counts.items()},
            "unique_groups": int(g["group_id"].nunique()) if "group_id" in g else None,
        }
    return stats


def split_summary(name: str, df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().sort_index().to_dict()
    named = {LABEL_NAMES.get(int(k), str(k)): int(v) for k, v in counts.items()}
    groups = int(df["group_id"].nunique()) if "group_id" in df.columns else None
    print(f"  {name:9s} n={len(df):6d}  groups={groups}  {named}")
    return {"n": int(len(df)), "groups": groups, "by_class": named}


def main():
    ap = argparse.ArgumentParser(description="PhishShield dataset builder v2 (leakage-resistant)")
    ap.add_argument("--csv", action="append", required=True,
                    help="Input CSV with text,label (repeatable)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--holdout-source", action="append", default=[],
                    help="Source name to hold out entirely as external/OOD test (repeatable)")
    ap.add_argument("--near-dup-threshold", type=float, default=0.85,
                    help="MinHash-estimated Jaccard threshold for near-duplicate merging. "
                         "Default 0.85 chosen by sweep on a 23.5k benchmark (20 templated "
                         "campaigns + 15.5k varied emails): 0.60 grouped campaigns perfectly but "
                         "over-merged distinct emails into 20 groups; 0.95 separated distinct "
                         "emails but fragmented campaigns into 15 groups (leakage risk); 0.85 "
                         "grouped campaigns into 3 while keeping 14,657 distinct emails separate. "
                         "Lower it if campaigns fragment across splits; raise it if distinct "
                         "emails are being merged.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-near-dup", action="store_true",
                    help="Skip near-duplicate clustering (exact dedup only) — faster, but leaky")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading sources...")
    df = load_csvs(args.csv)
    df = df[df["text"].notna()]
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"  loaded {len(df)} rows from {df['source'].nunique()} source(s)")

    # ---- Hold out external/OOD sources BEFORE any dedup or splitting ----
    external = pd.DataFrame()
    if args.holdout_source:
        mask = df["source"].astype(str).isin([str(s) for s in args.holdout_source])
        external = df[mask].reset_index(drop=True)
        df = df[~mask].reset_index(drop=True)
        print(f"  held out {len(external)} rows as external/OOD "
              f"(sources: {args.holdout_source})")

    # ---- Exact dedup on NORMALISED text ----
    before = len(df)
    df["_norm"] = df["text"].map(light_normalize)
    df = df.drop_duplicates(subset=["_norm"]).reset_index(drop=True)
    print(f"  normalised-exact dedup: removed {before - len(df)} rows -> {len(df)}")

    # ---- Near-duplicate clustering -> group_id ----
    if args.no_near_dup:
        df["group_id"] = range(len(df))
        print("  near-duplicate clustering SKIPPED (--no-near-dup): splits may leak")
    else:
        print(f"  clustering near-duplicates (threshold={args.near_dup_threshold})...")
        df["group_id"] = cluster_near_duplicates(
            df["text"].tolist(), threshold=args.near_dup_threshold)
        n_groups = df["group_id"].nunique()
        collapsed = len(df) - n_groups
        print(f"  -> {n_groups} groups ({collapsed} rows share a group with another)")

    df = df.drop(columns=["_norm"])

    # ---- Group-aware split ----
    cfg = SplitConfig(seed=args.seed)
    train, val, test = group_aware_split(df, cfg)

    print("\nSplits (group-aware — no template spans two splits):")
    stats = {
        "train": split_summary("train", train),
        "val": split_summary("val", val),
        "test": split_summary("test", test),
    }
    if len(external):
        external["group_id"] = -1
        stats["external"] = split_summary("external", external)

    # ---- Leakage assertion: no group_id appears in more than one split ----
    tr_g, va_g, te_g = set(train["group_id"]), set(val["group_id"]), set(test["group_id"])
    overlaps = {
        "train_val": len(tr_g & va_g),
        "train_test": len(tr_g & te_g),
        "val_test": len(va_g & te_g),
    }
    if any(overlaps.values()):
        print(f"\n  !! LEAKAGE DETECTED across splits: {overlaps}")
    else:
        print("\n  leakage check: PASS (no group appears in more than one split)")

    # ---- Write ----
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    if len(external):
        external.to_csv(out / "external_test.csv", index=False)

    with open(out / "label_map.json", "w") as f:
        json.dump(LABEL_NAMES, f, indent=2)

    manifest = {
        "builder": "v2-leakage-resistant",
        "seed": args.seed,
        "near_dup_threshold": None if args.no_near_dup else args.near_dup_threshold,
        "near_dup_clustering": not args.no_near_dup,
        "holdout_sources": args.holdout_source,
        "splits": stats,
        "leakage_check": {"overlaps": overlaps, "passed": not any(overlaps.values())},
        "per_source": per_source_stats(pd.concat([train, val, test], ignore_index=True)),
    }
    if len(external):
        manifest["per_source_external"] = per_source_stats(external)
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote train/val/test{'/external_test' if len(external) else ''} "
          f"+ label_map.json + manifest.json to {out.resolve()}")
    print("\nNOTE: metrics from this split will likely be LOWER than the v1 random "
          "split. That drop is the leakage that was previously being counted as skill.")


if __name__ == "__main__":
    main()