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

# A body heading has the number and title on one line: "Item 1A. Risk Factors".
# The table of contents is a table, so its rows look like "Item 1A. | Risk Factors | 5";
# requiring the title not to start with "|" skips them.
HEADING_RE = re.compile(r"^Item\s+(\d{1,2}[A-C]?)\.\s+([^|\s].*)$", re.IGNORECASE)

# Page footers like "Apple Inc. | 2025 Form 10-K | 20".
FOOTER_RE = re.compile(r"^.+\|\s*\d{4} Form 10-K\s*\|\s*\d+$")

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

    # Flatten each table row to "cell | cell | cell" so rows stay intact.
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", "%", ")"}]
        row.replace_with(soup.new_string("\n" + " | ".join(cells) + "\n"))

    lines = []
    for raw in soup.get_text("\n").splitlines():
        line = re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()
        if line and not FOOTER_RE.match(line):
            lines.append(line)
    return lines


def split_sections(lines: list[str]) -> list[Section]:
    """Group lines under their "Item N." heading. Text before the first heading is dropped."""
    sections: list[Section] = []
    current: Section | None = None
    body: list[str] = []

    for line in lines:
        match = HEADING_RE.match(line)
        if match and len(line) < 200:
            if current:
                current.text = "\n".join(body)
                sections.append(current)
            current = Section(item=match.group(1).upper(), title=match.group(2), text="")
            body = []
        elif current:
            body.append(line)

    if current:
        current.text = "\n".join(body)
        sections.append(current)
    return sections


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
