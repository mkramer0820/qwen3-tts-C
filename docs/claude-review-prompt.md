# Final Claude review prompt

Copy the text below into Claude while it is working in this repository.

```text
Perform a final code review of the current AMD-rocm branch as a senior C/C++,
ROCm/HIP, PyTorch, Docker, and Windows/WSL engineer. Review first: do not modify
files until you have reported findings and the repository owner approves fixes.

Repository checks

1. Run git branch --show-current, git status -sb, and git log -5 --oneline.
2. Confirm the branch is AMD-rocm and the working tree is clean.
3. Confirm upstream commit 328ab9c is an ancestor of HEAD.
4. The upstream merge is 7a74a0a. Commits c950a3a and e99b3d5 cancel each other.
5. Commit b05a1bb is the pre-hardening review-prompt baseline. Review the complete
   diff from b05a1bb to HEAD as well as interactions with the existing ROCm code.

Read first

- docs/rocm-review-handoff.md
- docs/amd-rocm.md
- docs/rocm-wsl-docker.md
- README.md

Required review areas

1. qwen_tts_rocm.cpp
   - Verify std::mutex construction and destruction are correct after replacing
     calloc/free with new/delete.
   - Verify the lock covers every mutation/use of the weight cache, dx/dy buffers,
     and hipBLAS handle without introducing recursion, deadlock, or bad fallback.
   - Recheck hipblasSgemm dimensions, memory layout, and every failure path.

2. Python training
   - Verify rocm_validation.py package checks and BF16 forward/backward probe.
   - Confirm qwen-tts 0.1.1, Transformers 4.57.3, Accelerate 1.12.0, and PEFT
     0.18.1 are mutually compatible with the exact APIs used by train_lora.py.
   - Independently validate main-talker label shifting and sub-talker hidden/codec
     alignment against the installed qwen-tts source.
   - Inspect the exact qwen-tts 0.1.1 and Transformers 4.57.3 wheels, not only the
     Qwen repository's current reference script. Reconcile any disagreement with
     the actual `text_embedding`, `text_projection`, and `ForCausalLMLoss` source.
   - Check finite loss/gradient handling under gradient accumulation and multiple
     Accelerator processes, including whether all workers fail consistently.
   - Check save/resume and .expr export behavior after the dependency pins.

3. Docker and WSL
   - Validate Dockerfile.rocm preserves matching ROCm Torch and torchaudio 2.9.1
     builds after pip and executes package/API smoke checks at build time.
   - Validate both Compose profiles; AMD changes must not alter CUDA defaults.
   - Confirm the CUDA Docker build context still includes requirements.txt and
     installs the shared pinned model stack without applying ROCm Torch checks.
   - Audit rocm-wsl.sh version matching, partial installations, dpkg package names,
     SHA-256 values, download URLs, quoting, and actionable error behavior.
   - Confirm AMD still maps librocdxg 1.2.0 to ROCm 7.2.x/R9700. Do not recommend
     a newer tag solely because it exists; require compatibility evidence.
   - Check /dev/dxg, libdxcore, librocdxg, dids.conf, Docker Engine selection,
     HIP_VISIBLE_DEVICES, gfx1201 detection, and OneDrive paths containing spaces.

4. Global repository compatibility
   - Confirm make rocm links Ingot and no removed safetensors reader is reachable.
   - Confirm CPU fallback bypasses GPU hooks and always writes valid output.
   - Confirm unavailable ROCm cannot produce a passing GPU self-test.
   - Check CPU, CUDA, and Metal code/build behavior for regressions.
   - Check voice cloning, voice-name metadata, --icl-only, 0.6B emotion assets,
     1.7B .expr, HTTP emotion, and quantization restrictions against documentation.

Evidence rules

- Do not assume an R9700 build, run, training job, clone, expression, audio test,
  or benchmark passed. Static validation is not hardware evidence.
- Keep all R9700 performance values Undetermined without raw target-machine logs.
- Clearly separate source review, syntax/config checks, compilation, container
  execution, model execution, and hardware results.

Run every safe test available. At minimum run git diff --check, Python compile or
AST checks, PowerShell parsing, bash -n, and Docker Compose validation. Compile and
test native targets only where the required toolchain exists. State every test not
run and why.

Output

1. Findings first, ordered critical/high/medium/low.
2. Include file:line, impact, reasoning or reproduction, and concrete fix.
3. Identify any finding from the earlier review that remains unresolved.
4. List assumptions and open questions.
5. List tests with pass/fail/not-run status.
6. End with approve, approve with follow-up, or changes required. Do not approve
   solely because documentation and static checks pass.
```
