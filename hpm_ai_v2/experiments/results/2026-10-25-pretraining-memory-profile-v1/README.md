# Source-only HFN pretraining memory profile

Two bounded prefixes of `data/pretraining/oasst1_train_en_1000.jsonl` were
trained with the same committed code revision (`9e621c51ea`) and a 10,000-node
hot cap. The profiler sampled process RSS every 100 ms and separately summed
the NumPy arrays retained in `TieredForest._mu_index`.

| Rows | Vocabulary | HFN nodes | Peak process RSS | Vector index | Elapsed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1,794 | 3,622 | 530.7 MiB | 50.7 MiB | 5.6 s |
| 300 | 3,919 | 9,034 | 1,368.7 MiB | 272.9 MiB | 13.1 s |

The `LazyPatternMap` retained zero resolved objects in both runs. This removes
the duplicate strong-reference map, but total training memory still grows:
the forest's vector index retains an array for every node, and its size grew
5.4× as node count grew 2.5× while vocabulary dimensionality more than doubled.
This is consistent with a node-count × vector-dimension cost; it is not a
memory-bounded training design.

These are resource measurements only, using the first 100 and 300 non-empty
rows, not source-grouped evaluation samples. They do not establish peak memory
for the full 1,000-row corpus or any larger dataset. A previous 1,000-row run
was stopped before completion, so its transient system-wide memory readings
are excluded from this report. Per-run input hashes, exact node/link counts,
RSS and code-file hashes are in `oasst100.json` and `oasst300.json`.

Rerun with, for example:

```bash
.venv/bin/python -m hpm_ai_v2.experiments.profile_pretrain_memory \
  --corpus data/pretraining/oasst1_train_en_1000.jsonl \
  --max-rows 300 \
  --output-dir /tmp/hpm-oasst-300 \
  --report /tmp/hpm-oasst-300-memory.json
```
