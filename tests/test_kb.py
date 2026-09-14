"""Knowledge-base provenance regression tests."""

from pathlib import Path

from app.kb import corpus_sha256, iter_kb_documents


def _policy(text: str) -> str:
    return f"""---
doc_id: demo
title: Demo policy
topic: general
---

## Example

{text}
"""


def test_readme_is_excluded_from_corpus_hash(tmp_path: Path):
    policy = tmp_path / "policy.md"
    readme = tmp_path / "README.md"
    policy.write_text(_policy("This policy text is long enough to represent indexed content."), encoding="utf-8")
    readme.write_text("Human documentation v1", encoding="utf-8")

    names = [p.name for p in iter_kb_documents(tmp_path)]
    before = corpus_sha256(tmp_path)

    assert names == ["policy.md"]

    readme.write_text("Human documentation v2", encoding="utf-8")
    assert corpus_sha256(tmp_path) == before

    policy.write_text(_policy("The indexed policy content has now changed."), encoding="utf-8")
    assert corpus_sha256(tmp_path) != before
