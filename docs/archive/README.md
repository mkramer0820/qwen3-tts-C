# Archived documentation

These files are kept for engineering history. They contain superseded plans,
abandoned methods, completed reviews, or branch-specific measurements. Do not use
them as current setup or support instructions. Start with the
[documentation index](../README.md).

For emotional TTS, the current source of truth is
[emotion-THE-recipe.md](../emotion-THE-recipe.md).

| file | what it was | why archived |
|---|---|---|
| `expressivity.md` | `--emotion` redefined as a `.vec`/compound-mood palette + control-vector how-to | superseded by the per-(voice×lang×emotion) recipe; the instruct+temperature finding lives in emotion-THE-recipe.md |
| `expressivity-recipes.md` | per-mood/per-language recipes on the `.vec` engine | the `.vec` control-vector method is abandoned; its per-language table contradicts the shipped recipe |
| `emotion-vector.md` | τ-vector (θ_emo − θ_neutral) task-arithmetic | abandoned (self-marked "NOT done"); replaced by CSP-FT |
| `emotion-seeds.md` | seed palette built on topk4 @ T0.8, EXPR-only, no steer | k4-era; contradicts the shipped k6 / T1.1 / steer recipe |
| `paralinguistics-ft-plan.md` | early para-FT blueprint | its root-cause diagnosis was overturned 2026-06-27 (BPE-split/special-token) |
| `roadmap-2026-06-04.md` | former root roadmap | replaced by the current R9700-focused `PLAN.md` |
| `gpu-accel-analysis-2026-07-02.md` | pre-implementation GPU design analysis | Metal/CUDA were later implemented and ROCm now has separate current docs |
| `gpu-accel-status-2026-07-02.md` | `feat/gpu-backends` status snapshot | branch and TODO state are no longer current |
| `perf-analysis-2026-07.md` | CPU optimization analysis | retained for reasoning; benchmark provenance moved to `performance.md` |
| `pr17-review-2026-07-10.md` | selective PR port review | completed historical review |
| `speculative-decoding-analysis.md` | unimplemented idea analysis | not on the current roadmap |
| `ingot-migration-validation-2026-08-02.md` | M1 migration validation | completed; not ROCm evidence |
| `cloned-voice-emotion-regression-2026-06-15.md` | old emotion regression investigation | superseded by current emotion and speaker guides |
| `multispeaker-emotion-pipeline-2026-06-16.md` | older dense/LoRA experiment pipeline | superseded by the pinned training guide |
| `quant-sub4-2026-07-14.md` | closed sub-4-bit quality study | quality gate failed; current support stops at INT4 |

Active docs: [emotion-THE-recipe.md](../emotion-THE-recipe.md),
[csp-ft-emotion.md](../csp-ft-emotion.md),
[expressivity-lora.md](../expressivity-lora.md),
[expressivity-assets.md](../expressivity-assets.md), and [markup.md](../markup.md).
