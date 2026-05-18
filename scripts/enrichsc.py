"""
enrich_sc.py
============
Enrich sc_matrimonial.parquet with signal flags extracted from
the description + raw_html columns that are already present.
No PDF download needed.

Run:
    python3 scripts/enrich_sc.py
    
Output:
    data/parquet/sc_enriched.parquet
    data/parquet/sc_enriched.csv
"""

import json, logging, re
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IN_PATH  = Path("data/parquet/sc_matrimonial.parquet")
OUT_PAR  = Path("data/parquet/sc_enriched.parquet")
OUT_CSV  = Path("data/parquet/sc_enriched.csv")

def find(pattern, text, flags=re.IGNORECASE):
    return bool(re.search(pattern, str(text), flags))

def extract_signals(row) -> dict:
    # Combine all available text
    text = " ".join(str(row.get(c, "")) for c in
                    ["description", "title", "petitioner", "respondent",
                     "raw_html", "disposal_nature"])

    # ── Case type ──────────────────────────────────────────────────────────────
    if find(r"quash|section.482|s\.482|482.*crpc", text):
        case_type = "quash"
    elif find(r"anticipatory.bail", text):
        case_type = "anticipatory_bail"
    elif find(r"\bbail\b", text):
        case_type = "bail"
    elif find(r"section.125|maintenance.*(?:wife|husband)|crpc.125", text):
        case_type = "maintenance"
    elif find(r"\bappeal\b|\brevision\b", text):
        case_type = "appeal"
    elif find(r"convict", text):
        case_type = "conviction"
    else:
        case_type = "other"

    # ── Outcome ────────────────────────────────────────────────────────────────
    disposal = str(row.get("disposal_nature_clean", "") or row.get("disposal_nature", "")).lower()
    title    = str(row.get("title", "")).lower()

    if find(r"quash", text) and find(r"allow", disposal + title):
        outcome = "quashed"
    elif find(r"allow|granted|partly allow", disposal):
        outcome = "allowed"
    elif find(r"dismiss|rejected", disposal):
        outcome = "dismissed"
    elif find(r"settl|comprom|mutual consent|disposed.*settl", text):
        outcome = "settled"
    elif "allow" in disposal:
        outcome = "allowed"
    elif "dismiss" in disposal:
        outcome = "dismissed"
    else:
        outcome = "unknown"

    # ── Statutes ───────────────────────────────────────────────────────────────
    statutes = []
    for pat, label in [
        (r"\b498\s*[aA]\b",           "IPC 498A"),
        (r"domestic.violence.act",     "DV Act"),
        (r"\b482\b.*crpc|crpc.*\b482\b","CrPC 482"),
        (r"\b125\b.*crpc|crpc.*\b125\b","CrPC 125"),
        (r"\b304\s*[bB]\b",           "IPC 304B"),
        (r"dowry.prohibition",         "Dowry Prohibition Act"),
        (r"hindu.marriage.act",        "Hindu Marriage Act"),
        (r"special.marriage.act",      "Special Marriage Act"),
    ]:
        if find(pat, text):
            statutes.append(label)

    # ── Signal flags ───────────────────────────────────────────────────────────
    flags = {
        "mediation_mentioned":         find(r"\bmediat", text),
        "settlement_mentioned":        find(r"\bsettl|\bcompromise\b|\bconciliat", text),
        "arrest_mentioned":            find(r"\barrest", text),
        "anticipatory_bail":           find(r"anticipatory.bail", text),
        "omnibus_vague_language":      find(r"omnibus|vague.allegation|sweeping.allegation|general.allegation", text),
        "relatives_accused":           find(r"mother.in.law|father.in.law|sister.in.law|brother.in.law|in.laws|relatives.of.(?:husband|accused)", text),
        "judicial_criticism_misuse":   find(r"misuse|abuse.of.process|legal.terrorism|weapon.*litigation|tool.*harass", text),
        "arnesh_kumar_cited":          find(r"arnesh.kumar", text),
        "rajesh_sharma_cited":         find(r"rajesh.sharma", text),
    }

    # ── Temporal ───────────────────────────────────────────────────────────────
    fir_m = re.search(r"fir.*?dated?\s+(\d{1,2}[\s\-\.]+\w+[\s\-\.]+\d{4})", text, re.IGNORECASE)
    mar_m = re.search(r"(?:married|marriage).*?(?:on|in)\s+(\w+\s+\d{4}|\d{4})", text, re.IGNORECASE)

    # ── Citations ──────────────────────────────────────────────────────────────
    cited = list(set(re.findall(
        r"AIR\s+\d{4}\s+\w+\s+\d+|\(\d{4}\)\s+\d+\s+SCC\s+\d+|\d{4}\s+\(\d+\)\s+SCC\s+\d+",
        text
    )))[:20]

    return {
        "case_type":             case_type,
        "outcome":               outcome,
        "statutes":              " | ".join(statutes),
        "cited_cases":           json.dumps(cited),
        "fir_date":              fir_m.group(1).strip() if fir_m else "",
        "marriage_date":         mar_m.group(1).strip() if mar_m else "",
        "word_count":            len(text.split()),
        **flags,
    }


def main():
    if not IN_PATH.exists():
        log.error(f"Not found: {IN_PATH}")
        return

    df = pd.read_parquet(IN_PATH)
    log.info(f"Loaded {len(df)} SC matrimonial cases")

    signals = df.apply(extract_signals, axis=1, result_type="expand")
    enriched = pd.concat([df, signals], axis=1)

    # Save
    enriched.to_parquet(OUT_PAR, index=False)
    enriched.drop(columns=["raw_html", "description"], errors="ignore").to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )
    log.info(f"Saved {len(enriched)} rows → {OUT_PAR}")

    # Summary
    print(f"\n{'='*55}")
    print(f"SC ENRICHED DATASET  ({len(enriched)} cases)")
    print(f"{'='*55}")
    print(f"\nCase types:\n{enriched['case_type'].value_counts().to_string()}")
    print(f"\nOutcomes:\n{enriched['outcome'].value_counts().to_string()}")
    print(f"\nStatutes (top 10):")
    from collections import Counter
    all_statutes = []
    for s in enriched["statutes"].dropna():
        all_statutes.extend([x.strip() for x in s.split("|") if x.strip()])
    for stat, cnt in Counter(all_statutes).most_common(10):
        print(f"  {stat:35s}: {cnt}")

    print(f"\nSignal flags (% of dataset):")
    flags = ["mediation_mentioned","settlement_mentioned","arrest_mentioned",
             "anticipatory_bail","omnibus_vague_language","relatives_accused",
             "judicial_criticism_misuse","arnesh_kumar_cited","rajesh_sharma_cited"]
    for f in flags:
        if f in enriched.columns:
            pct = enriched[f].mean() * 100
            bar = "█" * int(pct / 5)
            print(f"  {f:42s}: {pct:5.1f}% {bar}")

    print(f"\nYear distribution:\n{enriched['year'].value_counts().sort_index().to_string()}")

    print(f"\nCross-tab case_type × outcome:")
    ct = pd.crosstab(enriched["case_type"], enriched["outcome"])
    print(ct.to_string())

    print(f"\nArnesh Kumar cited: {enriched['arnesh_kumar_cited'].sum()} cases")
    print(f"Rajesh Sharma cited: {enriched['rajesh_sharma_cited'].sum()} cases")


if __name__ == "__main__":
    main()
