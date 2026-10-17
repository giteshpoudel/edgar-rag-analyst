"""Embed chunks and store them in LanceDB (an embedded vector database in data/lancedb)."""

from pathlib import Path

import lancedb

from edgar_rag import config
from edgar_rag.chunking import Chunk
from edgar_rag.embeddings import embed_documents

TABLE_NAME = "chunks"


def connect() -> lancedb.DBConnection:
    return lancedb.connect(str(config.data_dir() / "lancedb"))


def load_chunks(jsonl_path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in jsonl_path.read_text().splitlines()]


def index_chunks(chunks: list[Chunk]) -> int:
    """Embed chunks and write them to the table, replacing any earlier copy of the same filing."""
    vectors = embed_documents([c.embedding_text() for c in chunks])
    rows = [{**c.model_dump(), "vector": v.tolist()} for c, v in zip(chunks, vectors)]

    db = connect()
    if TABLE_NAME not in db.table_names():
        db.create_table(TABLE_NAME, data=rows)
    else:
        table = db.open_table(TABLE_NAME)
        # Re-indexing a filing replaces it instead of adding duplicates.
        for accession in {c.accession_number for c in chunks}:
            table.delete(f"accession_number = '{accession}'")
        table.add(rows)
    return len(rows)
