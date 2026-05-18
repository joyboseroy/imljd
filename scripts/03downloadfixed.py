"""
03_download_fixed.py
====================
Fixed based on actual schema discovered:

SC columns: title, petitioner, respondent, description, judge, author_judge,
            citation, case_id, cnr, decision_date, disposal_nature, court,
            available_languages, raw_html, path, nc_display, scraped_at, year

HC parquet: metadata/parquet/year=YYYY/court=XX/bench=YYY/metadata.parquet

Key fix: matrimonial terms are in raw_html (HTML snippets), not title alone.
         Search raw_html + title + petitioner + respondent combined.

Usage:
    # SC - search raw_html properly
    python3 scripts/03_download_fixed.py --sc --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024

    # Find Delhi HC court ID
    python3 scripts/03_download_fixed.py --find-delhi

    # HC parquet for specific court + year range
    python3 scripts/03_download_fixed.py --hc --court-id 29_3 --years 2018 2019 2020 2021 2022
    python3 scripts/03_download_fixed.py --hc --court-id 3_15 --years 2018 2019 2020 2021 2022
    python3 scripts/03_download_fixed.py --hc --court-id 27_1 --years 2018 2019 2020 2021 2022
    python3 scripts/03_download_fixed.py --hc --court-id 16_20 --years 2018 2019 2020 2021 2022

    # Run all priority courts
    python3 scripts/03_download_fixed.py --hc-all --years 2018 2019 2020 2021 2022 2023
"""

import argparse
import io
import json
import logging
import re
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HC_BUCKET = "indian-high-court-judgments"
SC_BUCKET = "indian-supreme-court-judgments"
S3_CFG    = Config(signature_version=UNSIGNED)
OUT_DIR   = Path("data/parquet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Priority HC courts — we'll discover Delhi ID via --find-delhi
HC_PRIORITY_COURTS = {
    "29_3":  "Karnataka HC",
    "27_1":  "Bombay HC",
    "16_20": "Allahabad HC (likely)",
    "19_16": "Calcutta HC",
    "3_22":  "Punjab Haryana HC",
    "32_4":  "Kerala HC",
}
# Delhi is probably 3_15 or similar — use --find-delhi to confirm

# Matrimonial terms — broader, for raw_html which contains full HTML snippets
MATRI_RE = re.compile(
    r"\b498\s*[aA]\b"
    r"|domestic.violence"
    r"|cruelty.{0,30}(?:husband|wife|matrimon)"
    r"|dowry"
    r"|matrimonial"
    r"|quash.{0,30}(?:fir|498|marriage)"
    r"|anticipatory.bail.{0,50}(?:498|husband|wife|matrimon)"
    r"|section.125.{0,30}(?:maintenance|wife)"
    r"|304\s*[bB]"
    r"|dowry.death"
    r"|cruelty.by.husband",
    re.IGNORECASE
)

# Broader fallback — for title/petitioner fields
MATRI_BROAD = re.compile(
    r"\b498[aA]?\b|dowry|matrimonial|cruelty|domestic.violence|maintenance|quash",
    re.IGNORECASE
)


def s3():
    return boto3.client("s3", config=S3_CFG, region_name="ap-south-1")


def download_parquet(client, bucket, key) -> pd.DataFrame:
    obj = client.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    return pd.read_parquet(buf)


# ── SC ─────────────────────────────────────────────────────────────────────────

def filter_sc(df: pd.DataFrame) -> pd.DataFrame:
    """
    SC columns include raw_html which has the full case card HTML.
    Search across: raw_html + title + petitioner + respondent + description
    """
    # Combine relevant text columns
    cols = ["raw_html", "title", "petitioner", "respondent", "description"]
    cols = [c for c in cols if c in df.columns]

    combined = df[cols].fillna("").astype(str).apply(
        lambda row: " ".join(row.values), axis=1
    )

    # Use broader regex on combined text
    mask = combined.apply(lambda t: bool(MATRI_BROAD.search(t)))
    filtered = df[mask].copy()
    filtered["_source"] = "sc"

    # Extract clean fields from raw_html
    if "raw_html" in filtered.columns:
        filtered["disposal_nature_clean"] = filtered["raw_html"].apply(
            lambda h: _extract_html_field(h, "Disposal Nature")
        )
        filtered["case_no_clean"] = filtered["raw_html"].apply(
            lambda h: _extract_html_field(h, "Case No")
        )

    return filtered


def _extract_html_field(html: str, field_name: str) -> str:
    """Extract a labelled field value from the SC raw_html snippets."""
    pattern = rf"{re.escape(field_name)}\s*:</span><font[^>]*>\s*([^<]+)"
    m = re.search(pattern, str(html), re.IGNORECASE)
    return m.group(1).strip() if m else ""


def run_sc(client, years: list[int]):
    frames = []
    first  = True

    for year in tqdm(sorted(years), desc="SC years"):
        key = f"metadata/parquet/year={year}/metadata.parquet"
        try:
            df = download_parquet(client, SC_BUCKET, key)
        except Exception as e:
            log.warning(f"SC {year}: {e}")
            continue

        if first:
            log.info(f"\nSC columns: {list(df.columns)}")
            # Print one sample raw_html to verify
            if "raw_html" in df.columns:
                sample = df["raw_html"].dropna().iloc[0] if len(df) else ""
                log.info(f"Sample raw_html (first 300 chars): {str(sample)[:300]}")
            first = False

        filtered = filter_sc(df)
        log.info(f"  SC {year}: {len(df)} total → {len(filtered)} matrimonial")
        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        log.warning("No SC matrimonial cases found — try broader years (2000-2024)")
        return

    combined = pd.concat(frames, ignore_index=True)
    # Drop raw_html from CSV (too large), keep in parquet
    out_parquet = OUT_DIR / "sc_matrimonial.parquet"
    out_csv     = OUT_DIR / "sc_matrimonial.csv"
    combined.to_parquet(out_parquet, index=False)
    combined.drop(columns=["raw_html"], errors="ignore").to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    log.info(f"\nSC matrimonial saved: {len(combined)} cases → {out_parquet}")
    print(f"\nYear distribution:\n{combined['year'].value_counts().sort_index().to_string()}")
    print(f"\nSample titles:")
    for t in combined["title"].dropna().head(10).tolist():
        print(f"  {t}")


# ── HC ─────────────────────────────────────────────────────────────────────────

def list_hc_benches(client, year: int, court_id: str) -> list[str]:
    """List bench subdirectories for a given court/year."""
    prefix = f"metadata/parquet/year={year}/court={court_id}/"
    r = client.list_objects_v2(Bucket=HC_BUCKET, Prefix=prefix, Delimiter="/")
    benches = [p["Prefix"] for p in r.get("CommonPrefixes", [])]
    # Also check for direct parquet files
    files = [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".parquet")]
    return benches, files


def download_hc_court_year(client, court_id: str, year: int) -> pd.DataFrame:
    """Download all bench parquet files for a court/year and combine."""
    benches, direct_files = list_hc_benches(client, year, court_id)

    frames = []

    # Direct parquet files at court level
    for key in direct_files:
        try:
            df = download_parquet(client, HC_BUCKET, key)
            df["_bench_key"] = key
            frames.append(df)
        except Exception as e:
            log.debug(f"  {key}: {e}")

    # Parquet files inside bench subdirectories
    for bench_prefix in benches:
        key = bench_prefix + "metadata.parquet"
        try:
            df = download_parquet(client, HC_BUCKET, key)
            df["_bench_key"] = key
            frames.append(df)
        except Exception as e:
            log.debug(f"  {key}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["_court_id"] = court_id
    combined["_year"]     = year
    combined["_source"]   = "hc"
    return combined


def filter_hc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Find all string columns to search
    str_cols = [c for c in df.columns
                if df[c].dtype == object and not c.startswith("_")]

    combined = df[str_cols].fillna("").astype(str).apply(
        lambda row: " ".join(row.values), axis=1
    )
    mask = combined.apply(lambda t: bool(MATRI_BROAD.search(t)))
    return df[mask].copy()


def run_hc(client, court_ids: list[str], years: list[int]):
    all_frames = []
    first_court = True

    for court_id in court_ids:
        court_name = HC_PRIORITY_COURTS.get(court_id, court_id)
        frames = []

        for year in sorted(years):
            df = download_hc_court_year(client, court_id, year)
            if df.empty:
                log.debug(f"  HC {court_name} {year}: no data")
                continue

            if first_court and not frames:
                log.info(f"\nHC columns ({court_name}): {list(df.columns)}")
                first_court = False

            filtered = filter_hc(df)
            log.info(f"  HC {court_name} {year}: {len(df)} → {len(filtered)} matrimonial")
            if not filtered.empty:
                frames.append(filtered)

        if frames:
            court_df = pd.concat(frames, ignore_index=True)
            safe_name = court_id.replace("/", "_")
            court_df.to_parquet(OUT_DIR / f"hc_{safe_name}.parquet", index=False)
            log.info(f"  Saved {len(court_df)} cases for {court_name}")
            all_frames.append(court_df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(OUT_DIR / "hc_matrimonial.parquet", index=False)
        combined.drop(columns=["raw_html"], errors="ignore").to_csv(
            OUT_DIR / "hc_matrimonial.csv", index=False, encoding="utf-8-sig"
        )
        log.info(f"\nHC matrimonial total: {len(combined)} cases")


# ── Find Delhi HC court ID ─────────────────────────────────────────────────────

def find_delhi(client):
    """
    List courts in year=2020 and download a tiny sample from each unknown court
    to find which one is Delhi HC.
    """
    log.info("Scanning all court IDs in year=2020 to identify Delhi HC...")
    r = client.list_objects_v2(
        Bucket=HC_BUCKET,
        Prefix="metadata/parquet/year=2020/",
        Delimiter="/",
        MaxKeys=50
    )
    unknown_courts = []
    for p in r.get("CommonPrefixes", []):
        court_id = p["Prefix"].split("court=")[-1].rstrip("/")
        if court_id not in HC_PRIORITY_COURTS:
            unknown_courts.append(court_id)

    print(f"\nChecking {len(unknown_courts)} unknown courts for Delhi HC...")

    for court_id in unknown_courts:
        benches, files = list_hc_benches(client, 2020, court_id)
        if not benches and not files:
            continue

        # Download first bench parquet
        key = (benches[0] + "metadata.parquet") if benches else files[0]
        try:
            df = download_parquet(client, HC_BUCKET, key)
            # Look for Delhi in any column
            str_cols = [c for c in df.columns if df[c].dtype == object]
            combined = df[str_cols].fillna("").astype(str).apply(
                lambda row: " ".join(row.values), axis=1
            )
            if combined.str.contains("delhi", case=False).any():
                print(f"  FOUND Delhi HC: court_id = {court_id}")
                print(f"    Bench: {key}")
                print(f"    Sample: {df.iloc[0][str_cols[0]] if str_cols else 'N/A'}")
                return court_id
            else:
                # Print first value to help identify
                sample_val = df.iloc[0][str_cols[0]] if str_cols and len(df) else "?"
                print(f"  {court_id}: {str(sample_val)[:60]}")
        except Exception as e:
            log.debug(f"  {court_id}: {e}")

    print("\nDelhi HC not found automatically. Review court IDs above.")
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc",         action="store_true")
    parser.add_argument("--hc",         action="store_true")
    parser.add_argument("--hc-all",     action="store_true",
                        help="Run all priority HC courts")
    parser.add_argument("--find-delhi", action="store_true")
    parser.add_argument("--court-id",   type=str, default="29_3")
    parser.add_argument("--years",      nargs="+", type=int,
                        default=list(range(2015, 2025)))
    args = parser.parse_args()

    client = s3()

    if args.find_delhi:
        find_delhi(client)
        return

    if args.sc:
        log.info(f"SC years: {args.years}")
        run_sc(client, args.years)
        return

    if args.hc:
        run_hc(client, [args.court_id], args.years)
        return

    if args.hc_all:
        court_ids = list(HC_PRIORITY_COURTS.keys())
        log.info(f"HC all priority courts: {court_ids}")
        run_hc(client, court_ids, args.years)
        return

    print("""
No action specified. Recommended sequence:

  # 1. SC with broader year range (raw_html now searched properly)
  python3 scripts/03_download_fixed.py --sc --years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024

  # 2. Find Delhi HC court ID
  python3 scripts/03_download_fixed.py --find-delhi

  # 3. HC for all known priority courts
  python3 scripts/03_download_fixed.py --hc-all --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024

  # 4. HC for specific court
  python3 scripts/03_download_fixed.py --hc --court-id 29_3 --years 2018 2019 2020 2021 2022 2023
""")


if __name__ == "__main__":
    main()
