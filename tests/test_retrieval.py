"""Keyword search query handling (W2 bug found in W5). Offline: a tiny LanceDB table."""

import lancedb
import pytest
from lancedb.index import FTS

from edgar_rag.retriever import fts_query


@pytest.fixture
def table(tmp_path):
    db = lancedb.connect(tmp_path)
    t = db.create_table("t", [{"id": 1, "text": "Depreciation, amortization, and other rose to 38,534"},
                              {"id": 2, "text": "Revenue grew"}])
    # Same index settings as index.build_keyword_index (no word positions).
    t.create_index("text", config=FTS(stem=True, remove_stop_words=True, language="English"), replace=True)
    return t


def test_quoted_query_fails_raw_and_works_after_cleaning(table):
    quoted = '"depreciation, amortization, and other"'
    with pytest.raises(ValueError, match="phrase queries"):
        table.search(quoted, query_type="fts").limit(5).to_list()
    hits = table.search(fts_query(quoted), query_type="fts").limit(5).to_list()
    assert [h["id"] for h in hits] == [1]


def test_curly_quotes_are_removed_too():
    assert '"' not in fts_query("“gross margin”") and "“" not in fts_query("“x”")
