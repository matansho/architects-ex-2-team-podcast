# Table description store

Shipped LLM (Gemma) descriptions for table chunks. ~3.6k JSON files.

Used by:

- `scripts/embed_corpus.py` — `--table-desc-cache` (default: this directory) applies
  `embed_text` + generator `payload` after Docling chunking, so teammates can rebuild
  the index **without** calling the describe API.
- `scripts/llm_describe_tables.py` — `--cache` defaults here; hits skip the API.

Regenerate (needs API key):

```bash
python scripts/llm_describe_tables.py --workers 8
```
