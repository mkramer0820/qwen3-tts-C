# Claude review prompt

Copy the text below into Claude while Claude is working in this repository.

```text
Review the current AMD-rocm branch as a senior C/C++, ROCm/HIP, PyTorch, Docker,
and Windows/WSL engineer. This is a code review first: do not modify files until
you have reported the findings and explained any proposed fixes.

Repository and history checks

1. Run:
   git branch --show-current
   git status -sb
   git log -3 --oneline --decorate
2. Confirm the branch is AMD-rocm and the working tree is clean.
3. The reviewed upstream merge is commit 7a74a0a. Commits c950a3a and e99b3d5
   cancel each other; e99b3d5 restores the files accidentally deleted by c950a3a.
   The current file tree should match 7a74a0a exactly.
4. Confirm gabriele-mastrapasqua/qwen3-tts main through upstream commit 328ab9c
   is an ancestor of the current branch.

Start by reading:

- docs/rocm-change-review-2026-08-18.md
- docs/upstream-sync-2026-08-18.md
- docs/amd-rocm.md
- docs/rocm-wsl-docker.md
- README.md

Review scope

1. Find correctness bugs, build failures, regressions, unsafe fallbacks, memory or
   lifetime problems, thread-safety problems, and missing error handling in the
   ROCm/HIP inference backend.
2. Verify the Ingot migration is complete for ROCm. In particular, confirm
   make rocm builds and links third_party/ingot/libingot.a and no removed legacy
   safetensors symbols remain reachable.
3. Review CPU fallback after HIP allocation, copy, or hipBLAS failures. Confirm
   fallback cannot recurse through the GPU hook and always produces valid output.
4. Review backend selection and --gpu-selftest. A requested but unavailable ROCm
   backend must return failure rather than comparing CPU against CPU and passing.
5. Review RDNA3/RDNA4 architecture handling, especially Radeon AI PRO R9700
   gfx1201. Do not broaden the requested hardware scope to MI/CDNA accelerators.
6. Review Windows 11, Ubuntu 24.04 WSL2, Docker Engine, Compose, /dev/dxg,
   librocdxg, GPU selection, paths containing spaces, and OneDrive-hosted checkout
   behavior. AMD support must remain opt-in; CUDA, Metal, and CPU paths must remain
   usable for the global repository.
7. Review PyTorch ROCm LoRA training: dependency pins, ROCm/BF16 guards, label
   alignment, codec tensor shape, text_projection coverage, save/resume, and .expr
   export compatibility.
8. Review voice cloning, .qvoice loading, metadata-only --voice-name behavior,
   --icl-only, 0.6B emotion assets and cloned-voice directions, 1.7B .expr behavior,
   HTTP emotion behavior, and INT4/quant-mixed restrictions.
9. Review the merged upstream server replay fix and check that the ROCm merge did
   not regress it.
10. Check documentation for commands or claims that do not match the code.

Evidence rules

- Do not assume ROCm hardware testing passed.
- No R9700 build, inference, training, cloning, expression, audio-quality, or
  performance result has been established merely by static checks.
- Keep R9700 performance values Undetermined unless supported by raw output from
  the actual target machine.
- Clearly separate source review, static validation, compilation, and hardware
  execution evidence.

Run every safe test available in the environment. At minimum, inspect the merge,
run git diff --check, parse Python and PowerShell, validate both Compose profiles,
and compile/test native targets where the required toolchains exist. State exactly
which tests could not run and why.

Output format

1. Findings first, ordered by severity (critical, high, medium, low).
2. For every finding, include file and line references, impact, reproduction or
   reasoning, and a concrete recommended fix.
3. List open questions and assumptions.
4. List tests run with pass/fail/not-run status.
5. Finish with a short verdict: approve, approve with follow-up, or changes
   required. Do not approve solely because documentation or static checks pass.
```
