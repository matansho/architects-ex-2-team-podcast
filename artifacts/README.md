# Table description store

Shipped LLM (Gemma) descriptions for table chunks. ~3.6k JSON files.

| Path | Role |
| --- | --- |
| `artifacts/table_descriptions/` | **Current** describe cache (prompt v2; applied by embed) |
| `artifacts/table_descriptions_v1/` | Optional local backup of pre–v2 descriptions (not required in git) |

Used by:

- `scripts/embed_corpus.py` — `--table-desc-cache` (default: this directory) applies
  `embed_text` + generator `payload` after Docling chunking, so teammates can rebuild
  the index **without** calling the describe API.
- `scripts/llm_describe_tables.py` — `--cache` defaults here; hits skip the API.
- `scripts/check_table_desc_quality.py` — quality gate (stop if bad).

Regenerate (needs API key; `--dry-run` skips index rewrite):

```bash
python scripts/llm_describe_tables.py --workers 8 --dry-run
python scripts/check_table_desc_quality.py
```
