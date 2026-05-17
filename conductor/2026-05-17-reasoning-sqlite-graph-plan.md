# Reasoning Graph to SQLite Plan

## Objective
Refactor the `ReasoningAgent` to perform graph traversal directly against the newly centralized SQLite `PatternStore`, eliminating the memory-intensive `_edge_index` and the 5-minute hanging bottleneck caused by pulling thousands of patterns into Python memory.

## Scope & Impact
- **Modifies:** `ReasoningAgent`, `PatternPager`.
- **Impact:** Eliminates the 5.0-minute hanging during `reason_with_trace`. Enables $O(1)$ memory graph traversal regardless of the total size of the global pattern cache.

## Implementation Steps

### 1. Add Graph Traversal Indexes to PatternPager
Update `PatternPager._init_schema()` to include:
```sql
CREATE INDEX IF NOT EXISTS idx_patterns_source ON patterns(source);
CREATE INDEX IF NOT EXISTS idx_patterns_target ON patterns(target);
```
This ensures high-speed lookups for outgoing and incoming reasoning edges directly at the database level.

### 2. Implement DB Edge Lookups in ReasoningAgent
Create the following methods to fetch edges on-demand:
- `_query_outgoing_edges(self, source_key: str) -> List[EdgeRecord]`
- `_query_incoming_edges(self, target_key: str, allowed_relations: set) -> List[EdgeRecord]`

These methods will:
1. Iterate through `self.reader.agents`.
2. Access `agent.pattern_pager._con`.
3. Execute `SELECT name, source, target, weight, dim FROM patterns WHERE source/target = ? ORDER BY weight DESC LIMIT X`.
4. Construct `EdgeRecord` objects on-the-fly, deriving the `relation` from the `agent_name`.

### 3. Implement On-Demand Cell Resolution
Refactor `_resolve_cell(self, token: str)` and `_choose_cell` to perform SQL `LIKE` queries across the databases to find relevant `source`/`target` keys instead of relying on a pre-built in-memory `_alias_index`.

### 4. Rip Out the In-Memory Graph
- **Delete:** `_edge_index`, `_node_index`, `_alias_index`.
- **Delete:** `_full_refresh()`, `_incremental_refresh()`, `_ensure_fresh()`.
- **Delete:** The state tracking variables `_dirty`, `_last_pattern_counts`, and `_refresh_state`.
- **Result:** The `ReasoningAgent` becomes completely stateless and purely analytical.

### 5. Add Ephemeral Query Caching
To prevent identical recursive calls from hitting the disk thousands of times during Abductive Search, wrap the `_query_outgoing_edges` and `_query_incoming_edges` methods with an ephemeral cache (e.g., `functools.lru_cache`) that is cleared at the start of each `reason_with_trace` call.

### 6. Verification & Test Fixes
- Create new unit tests specifically for the SQLite-backed graph traversal to catch regressions.
- Update `test_reasoning_agent.py` to ensure mock agents/stubs are compatible with the new direct-query architecture.
- Run the full test suite to guarantee zero regressions.
