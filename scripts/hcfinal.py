"""
hc_final.py
===========
Final HC extractor. Court-specific strategies:

  Karnataka 29_3: CRL prefix + (482)/matrimonial in description  [HIGH PRECISION]
  Delhi     7_26: CRL.M.C. + CRL.REV.P. + W.P.(CRL) only — no Pass 2 needed
                  (Delhi CRL.M.C. = quash petitions, overwhelmingly 498A)
  Rajasthan  8_9: CRLMP/CRLMB/CRLR + matrimonial in description
  Calcutta  19_16: CRR/CRA + matrimonial in description
  Kerala    32_4: Crl.MC/Crl.A + matrimonial in description

Run:
    python3 scripts/hc_final.py --courts 7_26 --years 2018 2019 2020 2021 2022 2023 2024
    python3 scripts/hc_final.py --courts 8_9  --years 2018 2019 2020 2021 2022 2023 2024
    python3 scripts/hc_final.py --all
"""

import argparse, io, logging, re
from pathlib import Path
import boto3, pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HC_BUCKET = "indian-high-court-judgments"
S3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="ap-south-1")
OUT = Path("data/parquet")
OUT.mkdir(parents=True, exist_ok=True)

HC_COURTS = {
    "29_3":  "Karnataka",
    "7_26":  "Delhi",
    "8_9":   "Rajasthan",
    "19_16": "Calcutta",
    "32_4":  "Kerala",
}

# Pass 1: criminal prefix per court
CRIMINAL_RE = {
    "29_3":  re.compile(r"\bCRL\b", re.IGNORECASE),
    "8_9":   re.compile(r"^(?:CRLMP|CRLMB|CRLR|CRLMC|CRLA)/", re.IGNORECASE),
    "19_16": re.compile(r"^(?:CRR|CRA|CRAN|CO)/", re.IGNORECASE),
    "32_4":  re.compile(r"^(?:Crl\.MC|Crl\.A|Crl\.RC|CRL\.MC)/", re.IGNORECASE),
    "7_26": re.compile(r"CRL\.M\.C\.", re.IGNORECASE),
}

# Courts where Pass 2 (description filter) is SKIPPED — take all criminal cases
# Delhi: description has no statute info, CRL.M.C. is reliably matrimonial
SKIP_PASS2 = set()  # empty now
DELHI_MATRI_RE = re.compile(r'CRL\.M\.C\..+Vs\s+STATE', re.IGNORECASE)

MATRI_RE = re.compile(
    r"\(482\)|498\s*[aA]|cruelty.*(?:husband|wife)|dowry"
    r"|domestic.violence|section\.?\s*125|304\s*[bB]"
    r"|matrimonial.offence|D\.?V\.?\s*Act",
    re.IGNORECASE
)


def list_benches(court_id: str, year: int) -> list[str]:
    prefix = f"metadata/parquet/year={year}/court={court_id}/"
    r = S3.list_objects_v2(Bucket=HC_BUCKET, Prefix=prefix, Delimiter="/")
    keys = [p["Prefix"] + "metadata.parquet" for p in r.get("CommonPrefixes", [])]
    keys += [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".parquet")]
    return keys


def download(key: str) -> pd.DataFrame:
    obj = S3.get_object(Bucket=HC_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def filter_cases(df: pd.DataFrame, court_id: str) -> pd.DataFrame:
    title = df["title"].astype(str)
    crl_re = CRIMINAL_RE.get(court_id, re.compile(r"\bCRL\b", re.IGNORECASE))
    crl_df = df[title.apply(lambda t: bool(crl_re.search(t)))]

    if crl_df.empty:
        return pd.DataFrame()

    if court_id == "7_26":
        mask = title.apply(lambda t: bool(DELHI_MATRI_RE.search(t)))
        result = df[mask].copy()
        # Sample to keep manageable — Delhi CRL.M.C. vs State includes non-498A
        return result.sample(min(len(result), 200), random_state=42) if len(result) > 200 else result

    if court_id in SKIP_PASS2:
        return crl_df.copy()

    desc = (crl_df["description"].astype(str) if "description" in crl_df.columns
            else pd.Series([""] * len(crl_df)))
    combined = desc + " " + crl_df["title"].astype(str)
    mask = combined.apply(lambda t: bool(MATRI_RE.search(t)))
    return crl_df[mask].copy()


def run_court(court_id: str, years: list[int], sample: bool = False) -> pd.DataFrame:
    court_name = HC_COURTS.get(court_id, court_id)
    frames = []

    for year in sorted(years):
        keys = list_benches(court_id, year)
        if not keys:
            continue

        year_frames = []
        for key in keys:
            try:
                df = download(key)
                df["_court_id"]   = court_id
                df["_court_name"] = court_name
                df["_year"]       = year
                df["_source"]     = "hc"
                df["_bench"]      = key.split("bench=")[-1].split("/")[0]
                filtered = filter_cases(df, court_id)
                if not filtered.empty:
                    year_frames.append(filtered)
            except Exception as e:
                log.warning(f"  {key}: {e}")

        if year_frames:
            year_df = pd.concat(year_frames, ignore_index=True)
            log.info(f"  {court_name} {year}: {len(year_df)}")
            frames.append(year_df)

            if sample:
                print(f"\n  Sample ({court_name} {year}):")
                for _, row in year_df.head(3).iterrows():
                    print(f"    {row.get('title','')[:70]}")
                    print(f"    {row.get('disposal_nature','')} | {str(row.get('description',''))[:80]}")
        else:
            log.info(f"  {court_name} {year}: 0")

    if not frames:
        return pd.DataFrame()

    court_df = pd.concat(frames, ignore_index=True)
    safe = court_id.replace("/", "_")
    court_df.to_parquet(OUT / f"hc_{safe}.parquet", index=False)
    log.info(f"  {court_name} total: {len(court_df)} → hc_{safe}.parquet")
    return court_df


def merge_all_courts():
    """Merge all per-court parquets into master hc_matrimonial.parquet."""
    frames = []
    for court_id, court_name in HC_COURTS.items():
        path = OUT / f"hc_{court_id}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Skip Punjab-Haryana (76k bloat) if present
            if court_id == "3_22":
                log.info(f"  Skipping Punjab-Haryana (too broad)")
                continue
            log.info(f"  {court_name}: {len(df)} cases")
            frames.append(df)

    if not frames:
        log.warning("No court parquets found")
        return

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["title", "_year"], keep="first")
    log.info(f"Dedup: {before} → {len(combined)}")

    combined.to_parquet(OUT / "hc_matrimonial.parquet", index=False)
    combined.drop(columns=["raw_html", "description"], errors="ignore").to_csv(
        OUT / "hc_matrimonial.csv", index=False,
        encoding="utf-8-sig", escapechar="\\"
    )

    print(f"\n{'='*55}")
    print(f"HC DATASET  ({len(combined)} cases)")
    print(f"{'='*55}")
    print(f"\nCourt breakdown:\n{combined['_court_name'].value_counts().to_string()}")
    print(f"\nYear breakdown:\n{combined['_year'].value_counts().sort_index().to_string()}")
    print(f"\nDisposal (top 12):\n{combined['disposal_nature'].value_counts().head(12).to_string()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--courts", nargs="+", default=None)
    parser.add_argument("--years",  nargs="+", type=int, default=list(range(2018, 2025)))
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--all",    action="store_true", help="Run all courts then merge")
    parser.add_argument("--merge",  action="store_true", help="Just merge existing parquets")
    args = parser.parse_args()

    if args.merge:
        merge_all_courts()
        return

    courts = list(HC_COURTS.keys()) if args.all else (args.courts or ["7_26"])

    for court_id in courts:
        run_court(court_id, args.years, args.sample)

    if args.all or len(courts) > 1:
        merge_all_courts()


if __name__ == "__main__":
    main()
