# sqlite-vec Pattern Store Design

## Goal

Replace `PatternPager`'s JSONL-based storage (index.jsonl, journal.jsonl, per-pattern JSON files) with a SQLite + sqlite-vec backend. Same public API, better scalability — no more unbounded file growth, no full-scan similarity search.

## Architecture

Each agent gets a single `patterns.db` SQLite file at:
```
<cache_dir>/<agent_name>/patterns.db
```

This replaces the entire `<cache_dir>/<agent_name>/` directory of files.

### Schema

```sql
CREATE TABLE IF NOT EXISTS patterns (
    rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT UNIQUE NOT NULL,
    dim     INTEGER NOT NULL,
    source  TEXT,
    target  TEXT,
    weight  REAL NOT NULL DEFAULT 1.0,
    embedding BLOB NOT NULL   -- float32 little-endian bytes
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_patterns USING vec0(
    embedding float[{dim}]    -- dim fixed per agent on first write
);
```

`vec_patterns.rowid` maps 1:1 to `patterns.rowid`. The `dim` is determined by the first pattern written and stored in a `meta` table:

```sql
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
-- INSERT OR IGNORE INTO meta VALUES ('dim', '<dim>');
```

If a pattern with a different dim is written, it is rejected (same behaviour as current `drop_incompatible_patterns`).

### WAL mode

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
```

Enables concurrent reads during writes. No compaction needed.

## Public API (unchanged)

`PatternPager.__init__(cache_dir, agent_name, max_active_patterns, spill_fraction)` — opens/creates `patterns.db`, loads `sqlite-vec` extension.

`save(cell)` / `enqueue_save(cell)` — background thread queue → `INSERT OR REPLACE INTO patterns` + matching upsert into `vec_patterns`.

`load(name, lookup)` — `SELECT * FROM patterns WHERE name = ?`.

`load_nearest(query_embedding, lookup, min_similarity, exclude_names)` — `SELECT rowid, distance FROM vec_patterns WHERE embedding MATCH ? AND k = 20`, join to `patterns`, cosine-filter to `min_similarity`.

`iter_index_payloads()` — `SELECT name, dim, source, target, weight, embedding FROM patterns` → list of dicts (same shape as before).

`load_from_payload(payload, lookup)` — unchanged (pure deserialisation, no I/O).

`select_evictions(patterns, weights)` — unchanged (pure logic, no I/O).

`has(name)` — `SELECT 1 FROM patterns WHERE name = ? LIMIT 1`.

`flush()` — `self._queue.join()` (same as now).

`close()` — flush + close db connection.

## Write Path

Background worker thread dequeues payloads and runs:

```python
with con:
    con.execute("INSERT OR REPLACE INTO patterns(name,dim,source,target,weight,embedding) VALUES (?,?,?,?,?,?)", ...)
    rowid = con.execute("SELECT rowid FROM patterns WHERE name=?", (name,)).fetchone()[0]
    con.execute("INSERT OR REPLACE INTO vec_patterns(rowid, embedding) VALUES (?,?)", (rowid, embedding_bytes))
```

All in one transaction. No journal file. No compaction.

## Migration

One-time migration script at `hpm_ai_v6/hpm_model/storage/migrate_pager_to_sqlite.py`:

1. For each agent directory under `cache_dir/`:
2. Read all `*.json` pattern files
3. Read `index.jsonl` for any additional entries
4. Insert all into new `patterns.db`
5. Print summary; leave old files in place (user deletes manually)

## Dependencies

```
sqlite-vec>=0.1.6
```

Added to `requirements.txt`. The extension is loaded via:
```python
import sqlite_vec
con.enable_load_extension(True)
sqlite_vec.load(con)
con.enable_load_extension(False)
```

## Files

- **Modify:** `hpm_ai_v6/hpm_model/storage/pattern_pager.py` — full rewrite of internals, same API
- **Create:** `hpm_ai_v6/hpm_model/storage/migrate_pager_to_sqlite.py` — one-time migration
- **Modify:** `requirements.txt` — add `sqlite-vec`
- **Create:** `hpm_ai_v6/tests/test_pattern_pager_sqlite.py` — tests for new backend

## Testing

- `test_save_and_load` — save a cell, load by name, verify fields
- `test_load_nearest` — save 3 cells with known embeddings, query nearest, verify closest returned
- `test_load_nearest_excludes` — excluded names not returned
- `test_iter_index_payloads` — returns all saved patterns as dicts
- `test_enqueue_save_async` — enqueue + flush, verify persisted
- `test_select_evictions` — excess patterns → correct indices returned
- `test_has` — True for saved name, False for unknown
- `test_duplicate_upsert` — saving same name twice updates weight, doesn't duplicate
- `test_incompatible_dim_rejected` — second pattern with different dim is dropped
- `test_migration_script` — writes JSON files, runs migration, verifies DB contents
