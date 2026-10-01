# Task curation funnel

Reconstructed end-to-end with verifiable evidence (file paths refer to the
authors' working repository; counts marked ✓ are re-derivable from artifacts
bundled in this release).

| stage | count | what happened |
|---|---|---|
| 0. imported generation-task universe | 742 | upstream families imported (CUDA samples 210, cudnn 113, CUTLASS 94, cublas 79, KernelBench L1 54 / L2 35 of 100 each, cutile 48, curand 34, cusolver 29, cusparse 27, cufft 7, KH class2 6, ThunderKittens 4, class3 2) |
| 1. generation-mode curation | 166 | quality/priority screen ("240 high-quality cases" collapsed to 166 configured) |
| 2. v1 debug panel | 166×5 | 5 source models attempt every task |
| 3. quality triage | 166 | trivial 2 / all-solved 2 / mixed 111 / none-solved 51 |
| 4. v2 exclusions | 145 | −21: 9 trivial-at-iter-1, 7 near-trivial (4/5 models), 5 infra-broken (audited per-task) |
| 5. broken-start selection | 145 | 1 per stem from real model iter-1 failures; 20–600 line window, anti-cheat static checks, category-quota balancing (144 auto + 1 manual) |
| 6. KernelHell backfill | 208 | +63 applied-family tasks (CUTLASS 20, class2 43) after metadata/compile screens |
| 7. v2 final list | 200 ✓ | −8 flash_attn head-dim/dtype variants |
| 8. v3 supplement | +16 | memory-crash instances harvested from v1 iterations |
| 9. paper corpus | 213 ✓ | 200 + 16 − 3 exact-duplicate broken starts |
| 10. this release | 228 ✓ | post-submission memory-crash expansion (+20/−4, incl. new cudabench family) and per-task fixes |

Notes on the record: per-source rejection counts for stage 0→1
and the intermediate 240-case artifact of stage 1 were not preserved; the
recorded reason for the 8 dropped variants at stage 7 was not; one
KernelHell class2 count (43 vs 44 in the design doc) is unexplained.

Solvability criterion: every kept task is **reference-solvable** (a working
reference implementation passes the testbench — see
`tests/test_family_fixtures.py`), audited as non-infrastructure-broken, and
**correctness-solvable**: at least one panel model produced a correct,
runnable repair with the performance gate disabled. Evaluation applies the
stricter performance gate on top of correctness, so the hardest tier remains
unsolved by every evaluated fixer under the default protocol (headroom).
