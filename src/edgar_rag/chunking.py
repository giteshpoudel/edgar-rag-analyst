"""Turn a 10-K HTML file into section-aware text chunks for retrieval.

Pipeline:  HTML -> clean lines -> Item sections -> overlapping chunks (with metadata)
"""

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pydantic import BaseModel

from edgar_rag.sec_client import Filing

# 10-Ks are XHTML; parsing them as HTML is fine and more forgiving.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# A body heading has the number and title on one line. Filers punctuate it differently:
# "Item 1A. Risk Factors", "Item 1A: Risk Factors" (Applied Materials), "ITEM 1.BUSINESS"
# (KLA, Broadcom). The table of contents is a table, so its rows look like
# "Item 1A. | Risk Factors | 5"; requiring the title not to start with "|" skips them.
HEADING_RE = re.compile(r"^Item\s+(\d{1,2}[A-C]?)\s*[.:]\s*([^|\s].*)$", re.IGNORECASE)

# Fallback for filers that don't number their sections (Intel uses its own layout plus a
# cross-reference index): standard titles, matched against the whole line.
TITLE_ITEMS = {
    "business": "1",
    "risk factors": "1A",
    "unresolved staff comments": "1B",
    "cybersecurity": "1C",
    "properties": "2",
    "legal proceedings": "3",
    "management's discussion and analysis": "7",
    "management's discussion and analysis of financial condition and results of operations": "7",
    "quantitative and qualitative disclosures about market risk": "7A",
    "financial statements and supplementary data": "8",
    "financial statements and supplemental details": "8",
    "controls and procedures": "9A",
    "exhibits": "15",
    "exhibits and financial statement schedules": "15",
}
MIN_NUMBERED_SECTIONS = 5  # fewer numbered headings than this -> try the title fallback

# Page footers like "Apple Inc. | 2025 Form 10-K | 20".
FOOTER_RE = re.compile(r"^.+\|\s*\d{4} Form 10-K\s*\|\s*\d+$")

# Elements that start a new line when rendered.
BLOCK_TAGS = ["p", "div", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "section"]

# Split long paragraphs at sentence ends.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Section(BaseModel):
    item: str  # "1A"
    title: str  # "Risk Factors"
    text: str


class Chunk(BaseModel):
    """One retrievable passage plus everything needed to cite it."""

    chunk_id: str
    ticker: str
    company: str
    form: str
    report_date: str
    accession_number: str
    source_url: str
    item: str
    section_title: str
    chunk_index: int
    text: str

    def embedding_text(self) -> str:
        """Text to embed: a short header gives each chunk context it lacks on its own."""
        return (
            f"{self.company} {self.form} ({self.report_date}), "
            f"Item {self.item}. {self.section_title}\n{self.text}"
        )


def html_to_lines(html: str) -> list[str]:
    """Extract readable text from a filing, one paragraph or table row per line."""
    soup = BeautifulSoup(html, "lxml")

    # Hidden XBRL metadata, not part of the readable document.
    for tag in soup(["ix:header", "script", "style"]):
        tag.decompose()

    # Break lines only at block elements, like a browser does. Inline elements (<span>, <font>)
    # are joined directly: some filings split one word across styled spans, e.g. Microsoft's
    # small-caps headings are "RIS" + "K FACTORS", and a separator between every text node
    # would cut words in half.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(BLOCK_TAGS):
        block.append("\n")

    # Flatten each table row to "cell | cell | cell" so rows stay intact.
    for row in soup.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text()).strip() for c in row.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", "%", ")"}]
        row.replace_with(soup.new_string("\n" + " | ".join(cells) + "\n"))

    lines = []
    for raw in soup.get_text().splitlines():
        line = re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()
        if line and not FOOTER_RE.match(line):
            lines.append(line)
    return lines


def split_sections(lines: list[str]) -> list[Section]:
    """Group lines under their "Item N." heading. Text before the first heading is dropped."""
    sections = _split(lines, _numbered_heading)
    if len({s.item for s in sections}) < MIN_NUMBERED_SECTIONS:
        by_title = _split(lines, _titled_heading)
        if len({s.item for s in by_title}) > len({s.item for s in sections}):
            sections = by_title
    return _merge_same_item(sections)


def _numbered_heading(line: str) -> tuple[str, str] | None:
    match = HEADING_RE.match(line)
    if match and len(line) < 200:
        return match.group(1).upper(), match.group(2)
    return None


def _titled_heading(line: str) -> tuple[str, str] | None:
    key = line.strip().rstrip(".").replace("’", "'").lower()
    item = TITLE_ITEMS.get(key)
    return (item, line.strip()) if item else None


def _split(lines: list[str], heading) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None
    body: list[str] = []
    for line in lines:
        found = heading(line)
        if found:
            if current:
                current.text = "\n".join(body)
                sections.append(current)
            current = Section(item=found[0], title=found[1], text="")
            body = []
        elif current:
            body.append(line)
    if current:
        current.text = "\n".join(body)
        sections.append(current)
    return sections


def _merge_same_item(sections: list[Section]) -> list[Section]:
    """A title can appear more than once (contents list, then body); keep one section per item
    so chunk ids (accession:item:index) stay unique. The first title is kept."""
    merged: dict[str, Section] = {}
    for s in sections:
        if s.item in merged:
            merged[s.item].text = "\n".join(t for t in (merged[s.item].text, s.text) if t)
        else:
            merged[s.item] = s.model_copy()
    return list(merged.values())


def chunk_text(text: str, max_chars: int = 1500, overlap_chars: int = 200) -> list[str]:
    """Pack paragraphs into chunks of up to max_chars, repeating a little text between chunks.

    Paragraph boundaries are kept where possible so a chunk doesn't start mid-thought.
    The overlap means a sentence that straddles two chunks is fully in at least one.
    """
    # Break any paragraph that is itself too long into sentence-sized pieces.
    pieces: list[str] = []
    for para in text.split("\n"):
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            pieces.extend(SENTENCE_RE.split(para))

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        if not piece.strip():
            continue
        # +1 per piece for the newline that joins them.
        if current and length + len(piece) + 1 > max_chars:
            chunks.append("\n".join(current))
            # Start the next chunk with the tail of this one (whole pieces only).
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(piece)
        length += len(piece) + 1

    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_filing(html: str, filing: Filing) -> list[Chunk]:
    """Full pipeline for one filing."""
    chunks: list[Chunk] = []
    for section in split_sections(html_to_lines(html)):
        for i, text in enumerate(chunk_text(section.text)):
            chunks.append(
                Chunk(
                    chunk_id=f"{filing.accession_number}:{section.item}:{i}",
                    ticker=filing.ticker,
                    company=filing.company,
                    form=filing.form,
                    report_date=filing.report_date,
                    accession_number=filing.accession_number,
                    source_url=filing.url,
                    item=section.item,
                    section_title=section.title,
                    chunk_index=i,
                    text=text,
                )
            )
    return chunks
