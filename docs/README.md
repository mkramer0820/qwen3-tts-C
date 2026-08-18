# Documentation index

This page separates current instructions from historical engineering records.
When a historical document conflicts with an active guide or the code, use the
active guide and verify behavior against the source.

## Start here

- [Repository README](../README.md): overview, quick start, capabilities, and
  clearly separated inherited benchmarks.
- [Current roadmap](../PLAN.md): outstanding R9700 hardware-validation gates.
- [Building](building.md): supported build targets and platform setup.
- [Performance](performance.md): inherited measurements and the unmeasured R9700
  table.

## AMD ROCm

- [Windows/WSL2 Docker workflow](rocm-wsl-docker.md): guided R9700 setup.
- [ROCm implementation](amd-rocm.md): C inference, PyTorch training, fallback, and
  limitations.
- [ROCm review handoff](rocm-review-handoff.md): reviewed changes, known risks,
  evidence boundary, and hardware acceptance checklist.
- [Independent review prompt](claude-review-prompt.md): reusable review procedure.

**Status:** source/static review is complete, but no R9700 hardware or container
execution has passed. Do not infer support, quality, or speed from configuration
checks.

## User guides

- [Voice cloning](voice-cloning.md), [custom voices](custom-voices.md), and
  [VoiceDesign](voice-design.md)
- [Emotion recipe](emotion-THE-recipe.md), [0.6B emotion](emotion-06b-recipe.md),
  [expressivity assets](expressivity-assets.md), and [inline markup](markup.md)
- [HTTP server](server.md) and [server batching](server-batching.md)
- [Quantization](quantization.md), [hardware testing](hardware-testing.md), and
  [x86 optimization](x86-optimization.md)

## Historical records

Superseded plans, experiment logs, rejected approaches, completed migration notes,
and old branch-specific reviews are retained in [archive](archive/README.md). They
are evidence and context, not current setup instructions.
