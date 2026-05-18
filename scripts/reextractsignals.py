"""
reextract_signals.py
====================
Re-extract all signals from cached txt files in data/extracted/.
Fixes the index misalignment where word_count>1000 rows have no signals.

Run:
    python3 scripts/reextract_signals.py
"""

import json, logging, re
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IN_PATH  = Path("data/parquet/sc_enriched.parquet")
TEXT_DIR = Path("data/extracted")


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


def main():
    df = pd.read_parquet(IN_PATH)
    log.info(f"Loaded {len(df)} cases")

    # Find all cached txt files
    txt_files = {f.stem.replace("sc_", ""): f for f in TEXT_DIR.glob("sc_*.txt")}
    log.info(f"Found {len(txt_files)} cached text files")

    updated = 0
    for idx, row in df.iterrows():
        path = str(row.get("path", ""))
        if not path or path not in txt_files:
            continue

        txt_file = txt_files[path]
        text = txt_file.read_text(encoding="utf-8", errors="replace")

        if len(text.split()) < 100:
            continue

        signals = extract_signals(text)
        for col, val in signals.items():
            df.at[idx, col] = val
        updated += 1

    log.info(f"Updated signals for {updated} cases")
    df.to_parquet(IN_PATH, index=False)
    log.info(f"Saved → {IN_PATH}")

    # Summary
    has_text = df["word_count"].fillna(0) > 1000
    sub = df[has_text]
    print(f"\n{'='*55}")
    print(f"SC DATASET  ({len(df)} cases, {len(sub)} with full text)")
    print(f"{'='*55}")

    flags = ["mediation_mentioned", "settlement_mentioned", "arrest_mentioned",
             "omnibus_vague_language", "relatives_accused",
             "judicial_criticism_misuse", "arnesh_kumar_cited", "rajesh_sharma_cited"]

    print("\nSignal flags (% of full-text cases):")
    for f in flags:
        if f in sub.columns:
            pct = sub[f].fillna(False).mean() * 100
            bar  = "█" * int(pct / 5)
            print(f"  {f:42s}: {pct:5.1f}% {bar}")

    print(f"\nAbsolute counts (all 1474 cases):")
    for f in flags:
        if f in df.columns:
            n = df[f].fillna(False).sum()
            print(f"  {f:42s}: {n}")

    # Cross-tab: judicial criticism x outcome (full text cases only)
    print(f"\nJudicial criticism x outcome (full text cases):")
    jc_sub = sub[sub["judicial_criticism_misuse"].fillna(False)]
    if len(jc_sub):
        print(jc_sub["outcome"].value_counts().to_string())

    print(f"\nArnesh Kumar cited in full-text cases: {sub['arnesh_kumar_cited'].fillna(False).sum()}")
    print(f"Omnibus language in full-text cases:   {sub['omnibus_vague_language'].fillna(False).sum()}")
    print(f"Mediation in full-text cases:          {sub['mediation_mentioned'].fillna(False).sum()}")


if __name__ == "__main__":
    main()
