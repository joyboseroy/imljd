"""
05_rag_interface.py
===================
RAG (Retrieval-Augmented Generation) query interface over the IMLJD corpus.

Two modes:
  --mode bm25     Simple BM25 keyword retrieval (no GPU, fast, good baseline)
  --mode hybrid   BM25 + embedding vector search (needs sentence-transformers)

Also: IRAC alignment mode — for a given case, applies the
falkor-irac Issue/Rule/Application/Conclusion structure.

Run:
    python scripts/05_rag_interface.py --mode bm25
    python scripts/05_rag_interface.py --mode bm25 --query "omnibus allegations quash"
    python scripts/05_rag_interface.py --irac --case-id abc123def456

Dependencies:
    pip install rank_bm25 sentence-transformers faiss-cpu (optional for hybrid)
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR    = Path("data")
EXTRACT_DIR = DATA_DIR / "extracted"
PARQUET     = DATA_DIR / "parquet" / "imljd_enriched.parquet"


# ── BM25 index ─────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self, df: pd.DataFrame):
        from rank_bm25 import BM25Okapi

        self.df = df.reset_index(drop=True)
        log.info("Building BM25 index...")

        # Corpus: combine key text fields
        corpus = []
        for _, row in df.iterrows():
            text_parts = [
                str(row.get("title",                "")),
                str(row.get("court",                "")),
                str(row.get("statutes",             "")),
                str(row.get("allegations_text",     ""))[:500],
                str(row.get("judicial_observations",""))[:500],
                str(row.get("case_type",            "")),
                str(row.get("outcome",              "")),
            ]
            corpus.append(" ".join(text_parts).lower().split())

        self.bm25 = BM25Okapi(corpus)
        log.info(f"BM25 index built: {len(corpus)} documents")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        results = []
        for i in top_indices:
            if scores[i] < 0.01:
                continue
            row = self.df.iloc[i]
            results.append({
                "score":        round(float(scores[i]), 3),
                "case_id":      str(row.get("case_id", "")),
                "title":        str(row.get("title", ""))[:80],
                "court":        str(row.get("court", ""))[:40],
                "year":         str(row.get("year", "")),
                "case_type":    str(row.get("case_type", "")),
                "outcome":      str(row.get("outcome", "")),
                "statutes":     str(row.get("statutes", "")),
                "mediation":    bool(row.get("mediation_mentioned", False)),
                "omnibus":      bool(row.get("omnibus_vague_language", False)),
                "relatives":    bool(row.get("relatives_accused", False)),
                "criticism":    bool(row.get("judicial_criticism_misuse", False)),
                "source_url":   str(row.get("source_url", "")),
            })

        return results


# ── IRAC extractor ─────────────────────────────────────────────────────────────

def extract_irac(row: pd.Series, full_text: str = "") -> dict:
    """
    Map judgment fields to IRAC structure.
    Aligns with falkor-irac pipeline at ~/irac/falkor-irac/

    Issue      → What legal question did the court address?
    Rule       → Which statutes / precedents apply?
    Application → How did the court reason?
    Conclusion → What was the outcome?
    """

    # Issue — infer from case_type + statutes
    statutes    = str(row.get("statutes", ""))
    case_type   = str(row.get("case_type", ""))
    title       = str(row.get("title", ""))

    issue_map = {
        "quash":             f"Whether FIR/proceedings under {statutes} should be quashed",
        "anticipatory_bail": f"Whether anticipatory bail should be granted in {statutes} case",
        "bail":              f"Whether bail should be granted under {statutes}",
        "maintenance":       "Whether and what quantum of maintenance is payable",
        "appeal":            f"Whether lower court order under {statutes} is sustainable",
        "conviction":        f"Whether conviction under {statutes} is sustainable",
        "other":             f"Legal question in matrimonial dispute — {title[:60]}",
    }
    issue = issue_map.get(case_type, issue_map["other"])

    # Rule — statutes + cited precedents
    cited_raw = str(row.get("cited_cases", "[]"))
    try:
        cited = json.loads(cited_raw) if cited_raw.startswith("[") else []
    except Exception:
        cited = []

    rule_parts = [f"Statutes: {statutes}"]
    if cited:
        rule_parts.append(f"Precedents cited: {'; '.join(cited[:5])}")
    rule = "\n".join(rule_parts)

    # Application — judicial observations
    application = str(row.get("judicial_observations", ""))

    # Enrich with signal flags
    flags = []
    if row.get("omnibus_vague_language"):      flags.append("omnibus/vague allegations noted")
    if row.get("relatives_accused"):           flags.append("extended family named as accused")
    if row.get("judicial_criticism_misuse"):   flags.append("judicial criticism of misuse")
    if row.get("mediation_mentioned"):         flags.append("mediation discussed")
    if row.get("arrest_mentioned"):            flags.append("arrest circumstances discussed")
    if flags:
        application += "\n\nKey judicial signals: " + "; ".join(flags)

    # Conclusion
    outcome = str(row.get("outcome", "unknown"))
    conclusion_map = {
        "quashed":   "Petition ALLOWED. FIR / proceedings quashed.",
        "allowed":   "Appeal / petition ALLOWED.",
        "dismissed": "Petition / appeal DISMISSED.",
        "settled":   "Matter SETTLED / disposed on compromise.",
        "unknown":   "Outcome not clearly determinable from available text.",
    }
    conclusion = conclusion_map.get(outcome, f"Outcome: {outcome}")

    return {
        "case_id":    str(row.get("case_id", "")),
        "title":      str(row.get("title", ""))[:100],
        "court":      str(row.get("court", ""))[:60],
        "year":       str(row.get("year", "")),
        "Issue":      issue,
        "Rule":       rule,
        "Application": application[:1500] if application else "Not extracted",
        "Conclusion": conclusion,
        "source_url": str(row.get("source_url", "")),
    }


def print_irac(irac: dict):
    print(f"\n{'='*60}")
    print(f"IRAC ANALYSIS")
    print(f"{'='*60}")
    print(f"Case:   {irac['title']}")
    print(f"Court:  {irac['court']} ({irac['year']})")
    print(f"URL:    {irac['source_url']}")
    print(f"\n[I] ISSUE\n{irac['Issue']}")
    print(f"\n[R] RULE\n{irac['Rule']}")
    print(f"\n[A] APPLICATION\n{irac['Application']}")
    print(f"\n[C] CONCLUSION\n{irac['Conclusion']}")
    print(f"{'='*60}")


# ── Analytics queries ──────────────────────────────────────────────────────────

def run_analytics(df: pd.DataFrame):
    """Pre-built analytical queries useful for research."""

    print(f"\n{'='*60}")
    print(f"IMLJD ANALYTICS ({len(df)} cases)")
    print(f"{'='*60}")

    print(f"\n1. Quash success rate by court:")
    quash_df = df[df["case_type"] == "quash"]
    if len(quash_df):
        rate = quash_df.groupby("court")["outcome"].apply(
            lambda x: (x == "quashed").sum() / len(x) * 100
        ).sort_values(ascending=False).head(10)
        print(rate.to_string())

    print(f"\n2. Mediation rate by case type:")
    med_rate = df.groupby("case_type")["mediation_mentioned"].mean() * 100
    print(med_rate.sort_values(ascending=False).to_string())

    print(f"\n3. Relatives accused — quash rate comparison:")
    for rel in [True, False]:
        sub = df[df["relatives_accused"] == rel]
        if len(sub) > 0:
            rate = (sub["outcome"] == "quashed").mean() * 100
            print(f"  Relatives accused={rel}: {rate:.1f}% quashed ({len(sub)} cases)")

    print(f"\n4. Omnibus allegations — quash rate:")
    for omn in [True, False]:
        sub = df[df["omnibus_vague_language"] == omn]
        if len(sub) > 0:
            rate = (sub["outcome"] == "quashed").mean() * 100
            print(f"  Omnibus language={omn}: {rate:.1f}% quashed ({len(sub)} cases)")

    print(f"\n5. Year trend (cases collected per year):")
    year_counts = df["year"].value_counts().sort_index()
    print(year_counts.tail(10).to_string())


# ── Interactive REPL ───────────────────────────────────────────────────────────

def repl(index: BM25Index, df: pd.DataFrame):
    print(f"\nIMJLD Search Interface ({len(df)} cases)")
    print("Commands: search <query> | irac <case_id> | analytics | quit\n")

    while True:
        try:
            cmd = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd or cmd in ("quit", "exit", "q"):
            break

        if cmd == "analytics":
            run_analytics(df)
            continue

        if cmd.startswith("irac "):
            case_id = cmd[5:].strip()
            rows = df[df["case_id"].astype(str) == case_id]
            if rows.empty:
                print(f"Case {case_id} not found")
            else:
                irac = extract_irac(rows.iloc[0])
                print_irac(irac)
            continue

        if cmd.startswith("search "):
            query = cmd[7:].strip()
        else:
            query = cmd

        results = index.search(query, top_k=8)
        if not results:
            print("No results found")
            continue

        print(f"\n{len(results)} results for '{query}':\n")
        for i, r in enumerate(results, 1):
            flags = []
            if r["omnibus"]:  flags.append("omnibus")
            if r["relatives"]: flags.append("relatives")
            if r["criticism"]: flags.append("misuse-criticism")
            if r["mediation"]: flags.append("mediation")
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            print(f"  {i}. [{r['score']:.2f}] {r['outcome']:10s} | "
                  f"{r['case_type']:18s} | {r['court'][:30]:30s} "
                  f"{r['year']}{flag_str}")
            print(f"     {r['title']}")
            print(f"     {r['source_url']}")
            print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--query",   type=str, default=None, help="One-shot query")
    parser.add_argument("--irac",    action="store_true", help="IRAC mode")
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--analytics", action="store_true")
    args = parser.parse_args()

    if not PARQUET.exists():
        log.error(f"Not found: {PARQUET}. Run 03_extract_text.py first.")
        return

    df = pd.read_parquet(PARQUET)
    log.info(f"Loaded {len(df)} enriched cases")

    if args.analytics:
        run_analytics(df)
        return

    if args.irac and args.case_id:
        rows = df[df["case_id"].astype(str) == args.case_id]
        if rows.empty:
            print(f"Case {args.case_id} not found")
        else:
            irac = extract_irac(rows.iloc[0])
            print_irac(irac)
        return

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.error("pip install rank_bm25")
        return

    index = BM25Index(df)

    if args.query:
        results = index.search(args.query, top_k=10)
        print(f"\n{len(results)} results for '{args.query}':\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.2f}] {r['outcome']:10s} | {r['court'][:35]} | {r['title']}")
        return

    # Interactive REPL
    repl(index, df)


if __name__ == "__main__":
    main()
