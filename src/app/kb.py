"""Knowledge-base loading and chunking.

Decision -> Reason -> Alternative -> Trade-off
- Decision: chunk on Markdown `## ` headings, then split oversized sections on
  paragraph boundaries with one paragraph of overlap.
- Reason: the corpus was written so that one section = one self-contained policy
  answer. Structural chunking keeps a fee table or an SLA list intact, which is
  exactly what a fixed 512-character window would cut in half.
- Alternative: fixed-size sliding window (simple, corpus-agnostic).
- Trade-off: structural chunking depends on the corpus being well-formed. For
  scraped or PDF content there are no reliable headings and a sliding window
  wins. Here we control the corpus, so we exploit it.

How to verify: `python -m training.build_index --dry-run` prints every chunk with
its length. No chunk should be a bare heading, and none should exceed MAX_CHARS.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 1200          # ~300 tokens: comfortably below any context budget
MIN_CHARS = 120           # anything shorter is merged into its neighbour
OVERLAP_PARAGRAPHS = 1


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    topic: str
    section: str
    text: str

    @property
    def indexable_text(self) -> str:
        """Title and section are repeated into the indexed text on purpose:
        a query like 'atm fee' should match the *heading* of the fee section."""
        return f"{self.title}. {self.section}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def iter_kb_documents(kb_dir: Path) -> list[Path]:
    """Return the canonical set of policy documents that make up the corpus.

    README files document the knowledge base for humans but are deliberately not
    indexed. Keeping this rule in one helper prevents the index builder, corpus
    hash, and runtime loader from drifting apart.
    """
    return [
        path
        for path in sorted(Path(kb_dir).glob("*.md"))
        if path.name.lower() != "readme.md"
    ]


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta: dict[str, str] = {}
    for line in raw[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[end + 4 :]


def _split_long_section(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        candidate = current + [para]
        if current and sum(len(p) for p in candidate) + 2 * len(candidate) > MAX_CHARS:
            parts.append("\n\n".join(current))
            current = current[-OVERLAP_PARAGRAPHS:] + [para] if OVERLAP_PARAGRAPHS else [para]
        else:
            current = candidate
    if current:
        parts.append("\n\n".join(current))
    return parts or [text]


def chunk_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    doc_id = meta.get("doc_id") or path.stem
    title = meta.get("title") or doc_id.replace("-", " ").title()
    topic = meta.get("topic") or "general"

    chunks: list[Chunk] = []
    # split on H2 headings, keeping the heading text
    pieces = re.split(r"^##\s+(.+)$", body, flags=re.MULTILINE)
    # pieces[0] is preamble (ignored), then alternating heading, content
    for i in range(1, len(pieces) - 1, 2):
        section = pieces[i].strip()
        content = pieces[i + 1].strip()
        if len(content) < MIN_CHARS and chunks:
            # merge a stub into the previous chunk rather than index a fragment
            prev = chunks.pop()
            merged = f"{prev.text}\n\n{section}\n{content}"
            chunks.append(
                Chunk(prev.chunk_id, prev.doc_id, prev.title, prev.topic, prev.section, merged)
            )
            continue
        for j, part in enumerate(_split_long_section(content)):
            suffix = "" if j == 0 else f"--{j + 1}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{slugify(section)}{suffix}",
                    doc_id=doc_id,
                    title=title,
                    topic=topic,
                    section=section,
                    text=part,
                )
            )
    return chunks


def load_corpus(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_kb_documents(kb_dir):
        chunks.extend(chunk_document(path))
    if not chunks:
        raise ValueError(f"no knowledge-base documents found in {kb_dir}")
    ids = [c.chunk_id for c in chunks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate chunk ids: {sorted(duplicates)}")
    return chunks
