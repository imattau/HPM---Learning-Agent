# Source-only HFN pretraining memory profile

Two bounded prefixes of `data/pretraining/oasst1_train_en_1000.jsonl` were
trained with the same committed code revision (`9e621c51ea`) and a 10,000-node
hot cap. The profiler sampled process RSS every 100 ms and separately summed
the NumPy arrays retained in `TieredForest._mu_index`.

| Rows | Vocabulary | HFN nodes | Peak process RSS | Vector index | Elapsed | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 100 | 1,794 | 3,622 | 530.7 MiB | 50.7 MiB | 5.6 s | complete |
| 300 | 3,919 | 9,034 | 1,368.7 MiB | 272.9 MiB | 13.1 s | complete |
| 1,000 | not completed | 22,298 | 6,154 MiB | 1,341 MiB | 44.5 s | stopped at 6 GiB RSS limit |

The `LazyPatternMap` retained zero resolved objects in both runs. This removes
the duplicate strong-reference map, but total training memory still grows:
the forest's vector index retains an array for every node, and its size grew
5.4× as node count grew 2.5× while vocabulary dimensionality more than doubled.
This is consistent with a node-count × vector-dimension cost; it is not a
memory-bounded training design.

These are resource measurements only, not source-grouped evaluation samples.
The 1,000-row run was explicitly stopped at the safety limit and its partial
checkpoint was deleted. At interruption, the hot cache was 10,003 nodes, the
term index had 4,846 terms and 46,001 links, and the forest index held 1.41 GB
of vectors. Sampled RSS overshot the 6 GiB trigger by 10.3 MiB during the
100 ms polling interval, so this is a software stop threshold, not a hard
operating-system memory limit. The abort report is `oasst1000-aborted-6gib.json`.
Per-run input hashes and code-file hashes are included in each JSON report.

Rerun with, for example:

```bash
.venv/bin/python -m hpm_ai_v2.experiments.profile_pretrain_memory \
  --corpus data/pretraining/oasst1_train_en_1000.jsonl \
  --max-rows 300 \
  --output-dir /tmp/hpm-oasst-300 \
  --report /tmp/hpm-oasst-300-memory.json \
  --max-rss-gib 6
```
