# Claims database and graph health (v3 Phase 3)

Built 2026-11-07 from extractor 990d655db2a7, linker efde3e3b501e, graph 80d33d6ba6df
(`edgar claims-load`, `edgar link`, `edgar graph`). Database: `data/v3/claims.sqlite` (local, not committed).

## Claims

- 75 filings, 3,580 claims loaded; all 3,580 quotes re-verified verbatim against the filing block at load.
- Every table rejects UPDATE and DELETE (SQLite triggers; tested in `tests/test_linking_graph.py`).

## Year-to-year links (60 company year pairs)

| Status | Rows |
|---|---|
| new | 1,304 |
| changed | 946 |
| unchanged | 716 |
| dropped | 1,119 |

How the 1,662 links were made: 626 identical quotes, 683 cosine >= 0.90, 353 decided by deepseek-flash
in the 0.72-0.90 band. Linking cost $0.063 list across three runs (two dev iterations and the full run);
rebuilding the database from the cache cost $0.

## Graph

| Edges | Count |
|---|---|
| states (company -> claim) | 3,580 |
| faces_risk (company -> risk topic, Item 1A claims) | 2,299 |
| has_segment | 204 |
| depends_on_supplier | 77 |
| names_customer | 38 |

Nodes: 36 companies (15 filers plus named customers and suppliers), 54 segments, 13 risk topics, 3,580 claims.
Entity aliases: `evals/v3/entity_aliases.csv` (33 reviewed merges).

Entity mentions that gave no edge: 76 whose name isn't in the supporting quote, 36 generic names
("Customer A", "single operating segment"), 8 naming the filer itself.

**Checks:** 0 edges without a verified quote; 0 entity edges whose name isn't in the supporting quote.

## Known errors

- Broadcom FY2021 `names_customer` Capital World Investors: the quote is about its largest *stockholder*.
  The extractor typed it `customer_concentration`; the edge is grounded but semantically wrong. Left in
  place (no hand edits to data); it's an extraction-precision error.
- Some supplier edges are loose: AMD -> Microsoft (driver certification), Microchip -> YMTC (a supplier's
  supplier). Grounded in the quote, typed by the extractor.
- Graph precision isn't measured by a labelled set; these come from reading all 31 distinct
  customer and supplier pairs (115 edges), and a few of their quotes.
