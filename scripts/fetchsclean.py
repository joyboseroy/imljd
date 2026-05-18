"""
fetch_sc_lean.py
================
Lean SC text fetcher:
- Only fetches years you specify
- Deletes tar after extracting (saves disk)
- Processes and saves incrementally
- Never hangs on massive tars

Recommended: fetch 2010-2024 only (post-CPC amendment era, 850+ cases)

Run:
    python3 scripts/fetch_sc_lean.py --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024
    python3 scripts/fetch_sc_lean.py --years 2014 2015 2016 2017 2018  # post-Arnesh Kumar only
    python3 scripts/fetch_sc_lean.py --cleanup-tars  # delete all cached tars to free space
"""

import argparse, json, logging, re, tarfile
from pathlib import Path
import boto3, pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SC_BUCKET = "indian-supreme-court-judgments"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="ap-south-1")

IN_PATH  = Path("data/parquet/sc_enriched.parquet")
TAR_DIR  = Path("data/tars")
TEXT_DIR = Path("data/extracted")
TAR_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


def get_tar_size_mb(year: str) -> float:
    """Check tar size before downloading."""
    key = f"data/tar/year={year}/english/english.tar"
    try:
        r = s3.head_object(Bucket=SC_BUCKET, Key=key)
        return r["ContentLength"] / 1e6
    except Exception:
        return 0


def extract_from_tar(tar_local: Path, paths: list[str]) -> dict[str, str]:
    """Extract multiple files from tar in one pass. Returns {path: text}."""
    results = {}
    path_set = set(paths)

    with tarfile.open(tar_local, "r") as tar:
        members = tar.getnames()
        for member in members:
            # Find which path this member matches
            matched = next((p for p in path_set if p in member), None)
            if not matched:
                continue
            try:
                f = tar.extractfile(member)
                if not f:
                    continue
                raw = f.read()
                for enc in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        results[matched] = raw.decode(enc)
                        break
                    except Exception:
                        pass
            except Exception as e:
                log.debug(f"Error extracting {member}: {e}")

            if len(results) == len(path_set):
                break  # got everything we need

    return results


def extract_signals(text: str) -> dict:
    def find(pat): return bool(re.search(pat, text, re.IGNORECASE))

    cited = list(set(re.findall(
        r"AIR\s+\d{4}\s+\w+\s+\d+|\(\d{4}\)\s+\d+\s+SCC\s+\d+", text
    )))[:20]

    alleg_m = re.search(
        r"(?:facts?|background|complainant\s+(?:alleged|stated))[:\s]*(.{100,1500}?)(?:\n{2,})",
        text, re.IGNORECASE | re.DOTALL
    )
    obs_m = re.search(
        r"(?:court\s+(?:observes?|notes?)|we\s+(?:note|observe|find))[:\s]*(.{100,1500}?)(?:\n{2,})",
        text, re.IGNORECASE | re.DOTALL
    )

    return {
        "mediation_mentioned":       find(r"\bmediat"),
        "settlement_mentioned":      find(r"\bsettl|\bcompromise\b"),
        "arrest_mentioned":          find(r"\barrest"),
        "omnibus_vague_language":    find(r"omnibus|vague.allegation|sweeping.allegation"),
        "relatives_accused":         find(r"mother.in.law|father.in.law|sister.in.law|in.laws"),
        "judicial_criticism_misuse": find(r"misuse|abuse.of.process|legal.terrorism"),
        "arnesh_kumar_cited":        find(r"arnesh.kumar"),
        "rajesh_sharma_cited":       find(r"rajesh.sharma"),
        "cited_cases":               json.dumps(cited),
        "allegations_text":          alleg_m.group(1).strip()[:1500] if alleg_m else "",
        "judicial_observations":     obs_m.group(1).strip()[:1500] if obs_m else "",
        "word_count":                len(text.split()),
        "full_text_preview":         text[:3000],
    }


def process_year(df: pd.DataFrame, year: str, delete_tar: bool = True) -> int:
    subset  = df[df["year"] == year]
    if subset.empty:
        return 0

    # Find which paths still need text
    needed = []
    for idx, row in subset.iterrows():
        path     = str(row.get("path", ""))
        txt_file = TEXT_DIR / f"sc_{path}.txt"
        if not txt_file.exists() and path:
            needed.append((idx, path))

    already_done = len(subset) - len(needed)
    if already_done:
        log.info(f"  {year}: {already_done} already extracted, {len(needed)} to fetch")

    if not needed:
        # Still update signals from existing txt files
        fetched = update_signals_from_cache(df, subset)
        return fetched

    # Check tar size
    tar_local = TAR_DIR / f"{year}_english.tar"
    if not tar_local.exists():
        size_mb = get_tar_size_mb(year)
        log.info(f"  {year}: downloading tar ({size_mb:.0f}MB)...")
        key = f"data/tar/year={year}/english/english.tar"
        s3.download_file(SC_BUCKET, key, str(tar_local))
        log.info(f"  {year}: downloaded ({tar_local.stat().st_size/1e6:.0f}MB)")

    # Extract all needed paths in one tar pass
    paths_only = [p for _, p in needed]
    log.info(f"  {year}: extracting {len(paths_only)} files from tar...")
    extracted = extract_from_tar(tar_local, paths_only)

    # Save text files and update df
    fetched = 0
    for idx, path in needed:
        text = extracted.get(path, "")
        if text and len(text.split()) > 100:
            txt_file = TEXT_DIR / f"sc_{path}.txt"
            txt_file.write_text(text, encoding="utf-8", errors="replace")
            signals = extract_signals(text)
            for col, val in signals.items():
                df.at[idx, col] = val
            fetched += 1

    # Delete tar to save space
    if delete_tar and tar_local.exists():
        tar_local.unlink()
        log.info(f"  {year}: tar deleted (space freed)")

    log.info(f"  {year}: {fetched} cases now have full text")

    # Save parquet incrementally after each year
    df.to_parquet(IN_PATH, index=False)

    return fetched


def update_signals_from_cache(df: pd.DataFrame, subset: pd.DataFrame) -> int:
    """Re-extract signals from already-cached txt files."""
    updated = 0
    for idx, row in subset.iterrows():
        path     = str(row.get("path", ""))
        txt_file = TEXT_DIR / f"sc_{path}.txt"
        if txt_file.exists():
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            if len(text.split()) > 100:
                signals = extract_signals(text)
                for col, val in signals.items():
                    df.at[idx, col] = val
                updated += 1
    return updated


def print_summary(df: pd.DataFrame):
    has_text = df["word_count"].fillna(0) > 1000
    sub = df[has_text]
    print(f"\n{'='*55}")
    print(f"SC DATASET  ({len(df)} cases, {len(sub)} with full text)")
    print(f"{'='*55}")

    flags = ["mediation_mentioned", "settlement_mentioned", "arrest_mentioned",
             "omnibus_vague_language", "relatives_accused",
             "judicial_criticism_misuse", "arnesh_kumar_cited", "rajesh_sharma_cited"]

    if len(sub):
        print("\nSignal flags (% of cases WITH full text):")
        for f in flags:
            if f in sub.columns:
                pct = sub[f].fillna(False).mean() * 100
                bar = "█" * int(pct / 5)
                print(f"  {f:42s}: {pct:5.1f}% {bar}")

    print(f"\nArnesh Kumar cited: {df['arnesh_kumar_cited'].fillna(False).sum()}")
    print(f"Omnibus language:   {df['omnibus_vague_language'].fillna(False).sum()}")
    print(f"Mediation:          {df['mediation_mentioned'].fillna(False).sum()}")
    print(f"Judicial criticism: {df['judicial_criticism_misuse'].fillna(False).sum()}")
    print(f"Relatives accused:  {df['relatives_accused'].fillna(False).sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=str,
                        default=[str(y) for y in range(2010, 2025)],
                        help="Years to process (default 2010-2024)")
    parser.add_argument("--keep-tars", action="store_true",
                        help="Don't delete tars after extraction")
    parser.add_argument("--cleanup-tars", action="store_true",
                        help="Delete all cached tars and exit")
    parser.add_argument("--summary", action="store_true",
                        help="Just print summary of current dataset")
    args = parser.parse_args()

    if args.cleanup_tars:
        tars = list(TAR_DIR.glob("*.tar"))
        total = sum(t.stat().st_size for t in tars) / 1e9
        for t in tars:
            t.unlink()
            log.info(f"Deleted {t.name}")
        log.info(f"Freed {total:.1f}GB")
        return

    df = pd.read_parquet(IN_PATH)
    log.info(f"Loaded {len(df)} cases")

    if args.summary:
        print_summary(df)
        return

    log.info(f"Processing years: {args.years}")
    log.info(f"Delete tars after extraction: {not args.keep_tars}")

    total = 0
    for year in args.years:
        n = process_year(df, year, delete_tar=not args.keep_tars)
        total += n

    print_summary(df)
    log.info(f"\nDone. Total cases with text extracted this run: {total}")

    # Clean CSV save
    drop_cols = ["raw_html", "description", "full_text_preview",
                 "allegations_text", "judicial_observations"]
    df.drop(columns=drop_cols, errors="ignore").to_csv(
        "data/parquet/sc_enriched.csv",
        index=False, encoding="utf-8-sig", escapechar="\\"
    )


if __name__ == "__main__":
    main()
