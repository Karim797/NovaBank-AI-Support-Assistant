# NovaBank Knowledge Base (FICTIONAL)

NovaBank is a **fictional** digital bank invented for this project. No policy,
fee, limit or SLA in these documents describes any real financial institution.
The content exists so the RAG pipeline has an internally consistent corpus to
retrieve from and so grounding/hallucination behaviour can be tested.

Conventions:
- One Markdown file per policy area; `doc_id` is stable and used as the citation key.
- `## ` headings are chunk boundaries — keep each section self-contained.
- Numbers (fees, limits, SLAs) are defined **once** in the owning document and
  referenced elsewhere by name, so the corpus cannot contradict itself.
- Base currency is GBP. NovaBank is UK-based, app-only, no branches.
