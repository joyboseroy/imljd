"""
fix_outcomes.py
===============
Fix SC outcome mapping using actual disposal_nature values found in data.
Also extracts judgment text from SC tar archives for a sample.

Run:
    python3 scripts/fix_outcomes.py               # fix outcomes only
    python3 scripts/fix_outcomes.py --fetch-text  # also fetch tar text (slow)
    python3 scripts/fix_outcomes.py --fetch-text --max 50
"""

import argparse, io, json, logging, re, tarfile
from pathlib import Path
from collections import Counter
import boto3, pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SC_BUCKET = "indian-supreme-court-judgments"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="ap-south-1")

IN_PATH  = Path("data/parquet/sc_enriched.parquet")
OUT_PAR  = Path("data/parquet/sc_enriched.parquet")
OUT_CSV  = Path("data/parquet/sc_enriched.csv")
TEXT_DIR = Path("data/extracted")
TEXT_DIR.mkdir(parents=True, exist_ok=True)

# ── Outcome mapping from actual disposal_nature values ─────────────────────────
# Based on: Appeal(s) allowed=663, Dismissed=380, Disposed off=232,
#           Case Partly allowed=95, Case Allowed=27, Leave Granted & Allowed=18

OUTCOME_MAP = {
    # Quash-specific — check case_type first
    # Allowed variants
    "appeal(s) allowed":          "allowed",
    "case allowed":               "allowed",
    "leave granted & allowed":    "allowed",
    "allowed":                    "allowed",
    "set aside":                  "allowed",       # lower court order set aside
    "case partly allowed":        "partly_allowed",
    # Dismissed variants
    "dismissed":                  "dismissed",
    "leave granted & dismissed":  "dismissed",
    # Disposed / settled
    "disposed off":               "disposed",      # often consent/settlement
    "leave granted & disposed off":"disposed",
    "directions issued":          "directions",
    "leave granted":              "leave_granted",
    "matter referred to larger bench": "referred",
    "reference answered":         "reference_answered",
    "hearing adjourned":          "adjourned",
}

def map_outcome(row) -> str:
    disposal = str(row.get("disposal_nature", "")).strip().lower()
    case_type = str(row.get("case_type", ""))
    
    base = OUTCOME_MAP.get(disposal, "unknown")
    
    # Refine: if case_type=quash and base=allowed → quashed
    if case_type == "quash" and base == "allowed":
        return "quashed"
    if case_type == "quash" and base == "partly_allowed":
        return "partly_quashed"
    # disposed_off for quash petitions often means settled
    if case_type == "quash" and base == "disposed":
        return "settled"
    
    return base


# ── SC tar text extraction ─────────────────────────────────────────────────────

def get_tar_text(year: str, path: str) -> str:
    """
    SC PDFs live inside yearly tar archives:
      data/tar/year=YYYY/english/english.tar
    
    The path column value (e.g. S_2000_2_114_134) is the filename
    inside the tar (as <path>.pdf or <path>.txt).
    
    Strategy: stream the tar, find the matching member, extract text.
    This downloads the whole tar (~50MB) so cache it locally.
    """
    tar_key   = f"data/tar/year={year}/english/english.tar"
    tar_local = Path(f"data/tars/{year}_english.tar")
    tar_local.parent.mkdir(parents=True, exist_ok=True)

    # Download tar if not cached
    if not tar_local.exists():
        log.info(f"Downloading tar: {tar_key} (may be ~50-100MB)")
        s3.download_file(SC_BUCKET, tar_key, str(tar_local))
        log.info(f"Downloaded: {tar_local}")

    # Extract matching file from tar
    target_names = [path, f"{path}.pdf", f"{path}.txt", f"{path}.PDF"]
    
    with tarfile.open(tar_local, "r") as tar:
        members = tar.getnames()
        # Find matching member
        match = next(
            (m for m in members if any(m.endswith(t) for t in target_names)),
            None
        )
        if not match:
            # Try partial match
            match = next((m for m in members if path in m), None)
        
        if not match:
            log.debug(f"  {path} not found in tar. Sample members: {members[:5]}")
            return ""
        
        f = tar.extractfile(match)
        if not f:
            return ""
        
        raw = f.read()
        
        # Try to decode as text
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return raw.decode(enc)
            except Exception:
                pass
        return raw.decode("utf-8", errors="replace")


def extract_signals_from_text(text: str) -> dict:
    """Re-run signal extraction on full judgment text."""
    def find(pat): return bool(re.search(pat, text, re.IGNORECASE))
    
    cited = list(set(re.findall(
        r"AIR\s+\d{4}\s+\w+\s+\d+|\(\d{4}\)\s+\d+\s+SCC\s+\d+",
        text
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
        "mediation_mentioned":        find(r"\bmediat"),
        "settlement_mentioned":       find(r"\bsettl|\bcompromise\b"),
        "arrest_mentioned":           find(r"\barrest"),
        "omnibus_vague_language":     find(r"omnibus|vague.allegation|sweeping.allegation"),
        "relatives_accused":          find(r"mother.in.law|father.in.law|sister.in.law|in.laws"),
        "judicial_criticism_misuse":  find(r"misuse|abuse.of.process|legal.terrorism"),
        "arnesh_kumar_cited":         find(r"arnesh.kumar"),
        "rajesh_sharma_cited":        find(r"rajesh.sharma"),
        "cited_cases":                json.dumps(cited),
        "allegations_text":           alleg_m.group(1).strip()[:1500] if alleg_m else "",
        "judicial_observations":      obs_m.group(1).strip()[:1500] if obs_m else "",
        "word_count":                 len(text.split()),
        "full_text_preview":          text[:3000],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-text", action="store_true",
                        help="Download SC tar archives and extract judgment text")
    parser.add_argument("--max", type=int, default=100,
                        help="Max cases to fetch text for (default 100)")
    parser.add_argument("--year", type=str, default=None,
                        help="Only fetch text for this year")
    args = parser.parse_args()

    df = pd.read_parquet(IN_PATH)
    log.info(f"Loaded {len(df)} cases")

    # ── Fix outcomes ────────────────────────────────────────────────────────────
    df["outcome"] = df.apply(map_outcome, axis=1)
    log.info("Outcomes remapped")

    # ── Optional: fetch full text from tar ─────────────────────────────────────
    if args.fetch_text:
        subset = df.copy()
        if args.year:
            subset = subset[subset["year"] == args.year]
        subset = subset.head(args.max)
        
        log.info(f"Fetching text for {len(subset)} cases")
        log.info("NOTE: This downloads ~50MB tar files per year. First run is slow.")
        
        text_signals = []
        years_downloaded = set()
        
        for _, row in subset.iterrows():
            path = str(row.get("path", ""))
            year = str(row.get("year", ""))
            
            if not path or not year:
                text_signals.append({})
                continue
            
            txt_file = TEXT_DIR / f"sc_{path}.txt"
            
            if txt_file.exists():
                text = txt_file.read_text(encoding="utf-8", errors="replace")
            else:
                try:
                    text = get_tar_text(year, path)
                    if text:
                        txt_file.write_text(text, encoding="utf-8", errors="replace")
                except Exception as e:
                    log.warning(f"  Error fetching {path}: {e}")
                    text = ""
            
            if text:
                sig = extract_signals_from_text(text)
                text_signals.append(sig)
                if year not in years_downloaded:
                    log.info(f"  Got text for {path} ({len(text.split())} words)")
                    years_downloaded.add(year)
            else:
                text_signals.append({})
        
        # Merge text signals back
        sig_df = pd.DataFrame(text_signals, index=subset.index)
        for col in sig_df.columns:
            df.loc[subset.index, col] = sig_df[col]
        
        log.info(f"Text fetched for {sum(1 for s in text_signals if s)} cases")

    # ── Save ────────────────────────────────────────────────────────────────────
    df.to_parquet(OUT_PAR, index=False)
    combined.drop(columns=["raw_html","description","full_text_preview"], errors="ignore").to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig", escapechar="\\"
    )
    log.info(f"Saved → {OUT_PAR}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"SC DATASET  ({len(df)} cases)")
    print(f"{'='*55}")
    print(f"\nCase types:\n{df['case_type'].value_counts().to_string()}")
    print(f"\nOutcomes (fixed):\n{df['outcome'].value_counts().to_string()}")
    
    if "word_count" in df.columns:
        with_text = df[df["word_count"] > 100]
        print(f"\nCases with full text: {len(with_text)}")
    
    print(f"\nSignal flags:")
    flags = ["mediation_mentioned","settlement_mentioned","arrest_mentioned",
             "omnibus_vague_language","relatives_accused",
             "judicial_criticism_misuse","arnesh_kumar_cited","rajesh_sharma_cited"]
    for f in flags:
        if f in df.columns:
            pct = df[f].mean() * 100
            print(f"  {f:42s}: {pct:.1f}%")

    print(f"\nCross-tab case_type × outcome:")
    print(pd.crosstab(df["case_type"], df["outcome"]).to_string())

    # Key research insight
    quash_df = df[df["case_type"] == "quash"]
    if len(quash_df):
        quash_success = (quash_df["outcome"] == "quashed").mean() * 100
        print(f"\nKey finding: {quash_success:.1f}% of 498A quashing petitions succeeded at SC level")
        print(f"({(quash_df['outcome']=='quashed').sum()} quashed out of {len(quash_df)} quash petitions)")


if __name__ == "__main__":
    main()
