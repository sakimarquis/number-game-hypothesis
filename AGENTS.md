# Agent Guide

## Read First

- `README.md` owns setup, commands, configuration entry points, cache safety, and output layout.
- `tasks/README.md` owns experiment terminology and task/evaluation semantics.
- `backends/README.md` owns backend capabilities, runtime behavior, and failure modes.
- `analysis/README.md` owns stage, artifact, CSV, and figure contracts.
- Consult local memory when available, then verify relevant claims against the live tree.

Keep responses concise, technical, and grounded in current code, configuration, caches, and artifacts.

## Working Boundary

- Investigate before asking questions or accepting assumptions about repository behavior.
- The primary agent owns repo-wide design and integration. Delegate only bounded evidence gathering or implementation, then review it against shared contracts.
- Before changing experiment semantics, public configuration, artifact formats, cache contracts, stage ownership, or shared architecture, present evidence, design, and trade-offs and wait for explicit approval.
- After approval, implement ordinary in-scope details without repeated confirmation.
- Preserve unrecognized worktree changes as other agents' work. Stop only when they materially block the requested change.
- Follow the cache safety rules in `README.md`; selector flags are write operations, not inspection tools.
- Update the document or module that owns a contract instead of duplicating it elsewhere.

## Implementation Principles

- Make the smallest root-cause change that preserves behavior and performance unless a trade-off is explicitly approved.
- Extend an existing seam before creating a parallel path. Keep task semantics in `tasks/`, backend behavior in `backends/`, reusable analysis in `analysis/`, and stage dispatch in the documented registries.
- Add abstractions only when they remove real duplication, preserve symmetry, adapt a representation or protocol, or own a shared lifecycle. Inline forwarding wrappers and single-use labels that add no policy.
- Let invariants fail at their lowest owner; avoid duplicate validation and impossible-case fallbacks. Retain explicit retries for unreliable external model/API outputs.
- Prefer readable code over cleverness. Keep semantic units together, make non-obvious shapes and formats explicit, and avoid one-argument-per-line formatting unless it conveys structure.
- Comment only non-obvious intent, equations, alignment rules, invariants, or failure modes. Comments describe the current contract, not change history.
- Use repository terminology from `tasks/README.md`. State assumptions and trade-offs rather than hiding uncertainty.
- Use GPU execution for model/tensor workloads when useful; keep pandas, SciPy, CSV, and plotting work on CPU unless measurement justifies otherwise.

## Verification

- Use `.venv\Scripts\python.exe` for Python and pytest in this checkout, or `uv run` when the environment must be created or refreshed.
- Match verification to risk: targeted tests for local changes, the full suite for shared-contract changes.
- For plotting-only changes, run the relevant analysis command and inspect the artifact; do not add plot snapshot tests.
- For documentation-only changes, verify links, symbols, commands, and described contracts against the live repository without running unrelated tests.
