"""Embed chunks and store them in LanceDB (an embedded vector database in data/lancedb)."""

from pathlib import Path

import lancedb

from edgar_rag import config
from edgar_rag.chunking import Chunk
from edgar_rag.embeddings import embed_documents

TABLE_NAME = "chunks"  # v2: the latest 10-K per company (the record; v3 never writes to it)
MULTI_TABLE_NAME = "chunks_multi"  # v3: 5 fiscal years per company, with a fiscal_year column


def connect() -> lancedb.DBConnection:
    return lancedb.connect(str(config.data_dir() / "lancedb"))


def load_chunks(jsonl_path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in jsonl_path.read_text().splitlines()]


def index_chunks(chunks: list[Chunk], table_name: str = TABLE_NAME, extra: dict | None = None,
                 keyword_index: bool = True) -> int:
    """Embed chunks and write them to a table, replacing any earlier copy of the same filing.

    extra: columns added to every row (v3 adds fiscal_year). keyword_index=False skips rebuilding
    the full-text index, so a batch can build it once at the end."""
    vectors = embed_documents([c.embedding_text() for c in chunks])
    rows = [{**c.model_dump(), **(extra or {}), "vector": v.tolist()} for c, v in zip(chunks, vectors)]

    db = connect()
    if table_name not in db.table_names():
        db.create_table(table_name, data=rows)
    else:
        table = db.open_table(table_name)
        # Re-indexing a filing replaces it instead of adding duplicates.
        for accession in {c.accession_number for c in chunks}:
            table.delete(f"accession_number = '{accession}'")
        table.add(rows)
    if keyword_index:
        build_keyword_index(table_name)
    return len(rows)


def indexed_count(accession: str, table_name: str) -> int:
    """Rows a filing has in a table (0 if the table doesn't exist yet)."""
    db = connect()
    if table_name not in db.table_names():
        return 0
    return db.open_table(table_name).count_rows(f"accession_number = '{accession}'")


def build_keyword_index(table_name: str = TABLE_NAME) -> None:
    """BM25 full-text index on chunk text, for keyword and hybrid search. Stemming lets
    "restructured" match "restructuring"; stop words ("the", "of") are ignored. Rebuilt after
    each indexing run; it takes well under a second for a few thousand chunks."""
    from lancedb.index import FTS

    table = connect().open_table(table_name)
    table.create_index("text", config=FTS(stem=True, remove_stop_words=True, language="English"), replace=True)
