"""
04_build_kg.py
==============
Build a knowledge graph from imljd_enriched.parquet.

Node types:   Case, Statute, Court, Outcome, AllegationType, Precedent
Edge types:   INVOKES, HEARD_BY, RESULTS_IN, CITES, ALLEGES, QUASHES

Two backends supported:
  --backend networkx   (default, no server needed, saves as GEXF + JSON)
  --backend neo4j      (requires running Neo4j instance)

Run:
    python scripts/04_build_kg.py
    python scripts/04_build_kg.py --backend neo4j --uri bolt://localhost:7687

Outputs (networkx):
    data/kg/imljd_graph.gexf      - for Gephi visualisation
    data/kg/imljd_graph.json      - node/edge list JSON
    data/kg/kg_stats.txt          - graph statistics
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = Path("data")
PARQUET    = DATA_DIR / "parquet" / "imljd_enriched.parquet"
KG_DIR     = DATA_DIR / "kg"
KG_DIR.mkdir(parents=True, exist_ok=True)


# ── NetworkX backend ───────────────────────────────────────────────────────────

def build_networkx_graph(df: pd.DataFrame):
    import networkx as nx

    G = nx.MultiDiGraph()

    # Add all nodes first
    # Case nodes
    for _, row in df.iterrows():
        cid = str(row["case_id"])
        G.add_node(cid,
            type="Case",
            label=str(row.get("title", cid))[:60],
            court=str(row.get("court", "")),
            year=str(row.get("year", "")),
            state=str(row.get("state", "")),
            case_type=str(row.get("case_type", "")),
            outcome=str(row.get("outcome", "")),
            mediation=bool(row.get("mediation_mentioned", False)),
            relatives_accused=bool(row.get("relatives_accused", False)),
            omnibus=bool(row.get("omnibus_vague_language", False)),
            judicial_criticism=bool(row.get("judicial_criticism_misuse", False)),
        )

    # Add edges and referenced nodes
    statute_counts  = defaultdict(int)
    court_counts    = defaultdict(int)
    outcome_counts  = defaultdict(int)

    for _, row in df.iterrows():
        cid = str(row["case_id"])

        # INVOKES statute edges
        statutes_str = str(row.get("statutes", ""))
        for statute in statutes_str.split(" | "):
            statute = statute.strip()
            if not statute:
                continue
            stat_node = f"STATUTE:{statute}"
            if not G.has_node(stat_node):
                G.add_node(stat_node, type="Statute", label=statute)
            G.add_edge(cid, stat_node, type="INVOKES")
            statute_counts[statute] += 1

        # HEARD_BY court edges
        court = str(row.get("court", "")).strip()
        if court:
            court_node = f"COURT:{court[:50]}"
            if not G.has_node(court_node):
                G.add_node(court_node, type="Court", label=court[:50])
            G.add_edge(cid, court_node, type="HEARD_BY")
            court_counts[court] += 1

        # RESULTS_IN outcome edges
        outcome = str(row.get("outcome", "unknown")).strip()
        out_node = f"OUTCOME:{outcome}"
        if not G.has_node(out_node):
            G.add_node(out_node, type="Outcome", label=outcome)
        G.add_edge(cid, out_node, type="RESULTS_IN")
        outcome_counts[outcome] += 1

        # CITES precedent edges
        cited_raw = str(row.get("cited_cases", "[]"))
        try:
            cited_list = json.loads(cited_raw) if cited_raw.startswith("[") else []
        except Exception:
            cited_list = []

        for cite in cited_list[:10]:
            cite = str(cite).strip()
            if not cite:
                continue
            prec_node = f"PRECEDENT:{cite}"
            if not G.has_node(prec_node):
                G.add_node(prec_node, type="Precedent", label=cite)
            G.add_edge(cid, prec_node, type="CITES")

        # Allegation type nodes
        case_type = str(row.get("case_type", "other"))
        alleg_node = f"ALLEGATION:{case_type}"
        if not G.has_node(alleg_node):
            G.add_node(alleg_node, type="AllegationType", label=case_type)
        G.add_edge(cid, alleg_node, type="ALLEGES")

        # If quashed — explicit QUASHES edge
        if outcome == "quashed":
            G.add_edge(cid, out_node, type="QUASHES")

    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, {"statutes": statute_counts, "courts": court_counts, "outcomes": outcome_counts}


def save_networkx(G, stats: dict):
    import networkx as nx

    # GEXF (Gephi compatible)
    gexf_path = KG_DIR / "imljd_graph.gexf"
    nx.write_gexf(G, str(gexf_path))
    log.info(f"GEXF saved: {gexf_path}")

    # JSON (node-link format for D3 / web)
    data = nx.node_link_data(G)
    json_path = KG_DIR / "imljd_graph.json"
    json_path.write_text(json.dumps(data, default=str), encoding="utf-8")
    log.info(f"JSON saved: {json_path}")

    # Stats
    stats_txt = KG_DIR / "kg_stats.txt"
    lines = [
        f"Nodes: {G.number_of_nodes()}",
        f"Edges: {G.number_of_edges()}",
        "",
        "Node types:",
    ]
    type_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        type_counts[data.get("type", "unknown")] += 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {t:20s}: {c}")

    lines += ["", "Top statutes invoked:"]
    for s, c in sorted(stats["statutes"].items(), key=lambda x: -x[1])[:10]:
        lines.append(f"  {s:30s}: {c}")

    lines += ["", "Outcomes:"]
    for o, c in sorted(stats["outcomes"].items(), key=lambda x: -x[1]):
        lines.append(f"  {o:20s}: {c}")

    stats_txt.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Stats: {stats_txt}")
    print("\n".join(lines))


# ── Neo4j backend ──────────────────────────────────────────────────────────────

def build_neo4j(df: pd.DataFrame, uri: str, user: str, password: str):
    """
    Load graph into Neo4j.
    Requires: pip install neo4j
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_nodes_and_edges(tx, row):
        cid = str(row["case_id"])

        # Case node
        tx.run("""
            MERGE (c:Case {case_id: $cid})
            SET c.title = $title,
                c.court = $court,
                c.year = $year,
                c.state = $state,
                c.case_type = $case_type,
                c.outcome = $outcome,
                c.mediation = $mediation,
                c.relatives_accused = $relatives_accused,
                c.omnibus = $omnibus,
                c.judicial_criticism = $judicial_criticism
        """, cid=cid,
             title=str(row.get("title", ""))[:200],
             court=str(row.get("court", "")),
             year=str(row.get("year", "")),
             state=str(row.get("state", "")),
             case_type=str(row.get("case_type", "")),
             outcome=str(row.get("outcome", "")),
             mediation=bool(row.get("mediation_mentioned", False)),
             relatives_accused=bool(row.get("relatives_accused", False)),
             omnibus=bool(row.get("omnibus_vague_language", False)),
             judicial_criticism=bool(row.get("judicial_criticism_misuse", False)),
        )

        # Statutes
        for statute in str(row.get("statutes", "")).split(" | "):
            statute = statute.strip()
            if statute:
                tx.run("""
                    MERGE (s:Statute {name: $name})
                    WITH s
                    MATCH (c:Case {case_id: $cid})
                    MERGE (c)-[:INVOKES]->(s)
                """, name=statute, cid=cid)

        # Court
        court = str(row.get("court", "")).strip()
        if court:
            tx.run("""
                MERGE (ct:Court {name: $court})
                WITH ct
                MATCH (c:Case {case_id: $cid})
                MERGE (c)-[:HEARD_BY]->(ct)
            """, court=court[:100], cid=cid)

        # Outcome
        outcome = str(row.get("outcome", "unknown"))
        tx.run("""
            MERGE (o:Outcome {type: $outcome})
            WITH o
            MATCH (c:Case {case_id: $cid})
            MERGE (c)-[:RESULTS_IN]->(o)
        """, outcome=outcome, cid=cid)

        # Citations
        cited_raw = str(row.get("cited_cases", "[]"))
        try:
            cited_list = json.loads(cited_raw) if cited_raw.startswith("[") else []
        except Exception:
            cited_list = []
        for cite in cited_list[:10]:
            cite = str(cite).strip()
            if cite:
                tx.run("""
                    MERGE (p:Precedent {citation: $cite})
                    WITH p
                    MATCH (c:Case {case_id: $cid})
                    MERGE (c)-[:CITES]->(p)
                """, cite=cite, cid=cid)

    with driver.session() as session:
        for _, row in df.iterrows():
            session.write_transaction(create_nodes_and_edges, row)

    log.info(f"Neo4j: loaded {len(df)} cases")
    driver.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",  choices=["networkx", "neo4j"], default="networkx")
    parser.add_argument("--uri",      default="bolt://localhost:7687")
    parser.add_argument("--user",     default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--max",      type=int, default=None)
    args = parser.parse_args()

    if not PARQUET.exists():
        log.error(f"Not found: {PARQUET}. Run 03_extract_text.py first.")
        return

    df = pd.read_parquet(PARQUET)
    log.info(f"Loaded {len(df)} enriched cases")

    if args.max:
        df = df.head(args.max)

    if args.backend == "networkx":
        try:
            import networkx
        except ImportError:
            log.error("pip install networkx")
            return
        G, stats = build_networkx_graph(df)
        save_networkx(G, stats)
        log.info("\nNext: python scripts/05_rag_interface.py")

    elif args.backend == "neo4j":
        build_neo4j(df, args.uri, args.user, args.password)
        log.info("\nNeo4j graph loaded. Open http://localhost:7474 to explore.")


if __name__ == "__main__":
    main()
