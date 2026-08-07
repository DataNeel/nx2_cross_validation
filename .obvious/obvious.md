# Obvious repo guidance

<!-- obvious-install: skill=autobuild-setup, skill-version=1.0.3, template-version=1 -->

This repo uses `.obvious/` for reviewed Autobuild guidance.

Before editing, suggest the smallest relevant set of `.obvious` files for the task. Match candidate files by reading their frontmatter.

## Codebase Map

Single-file Python library — no subdirectories, no build system, no test suite.

| Path | Purpose |
|---|---|
| `Nx2cv.py` | Main library — N×2 cross-validation for comparing binary classifiers on imbalanced datasets |
| `README.md` | Usage documentation with API reference and example |

## Repo Guidance for Autobuild

No AGENTS.md, CLAUDE.md, CONTRIBUTING.md, .cursorrules, or .windsurfrules found. Guidance synthesized from README.md and source code.

### Stack

- **Language:** Python 3 (script uses deprecated pandas `.ix` API — incompatible with pandas >=0.21)
- **Dependencies:** pandas, numpy, scipy, scikit-learn (no requirements.txt or pyproject.toml — install manually)
- **No build system, no test framework, no linter configured**

### Running

```bash
python3 Nx2cv.py
```

**Warning:** The script fails on modern pandas (>=0.21) due to use of removed `.ix` indexer and `random.sample(X.index, ...)` incompatibility. The demo section at the bottom generates a 100K-row synthetic dataset and runs 6 classifiers — expect long execution times even if API issues are fixed.

### Conventions

- Single-file library; all functions and the demo script live in `Nx2cv.py`
- Models must have a `.name` attribute set before being passed to `evaluate_models()`
- Uses Python `multiprocessing.Pool` for parallel model evaluation

## Local Verification

> **Warning:** Running full-repo typecheck, lint, or tests may OOM or timeout in the sandbox for large repos.
> Use the scoped commands below when verifying changes.

### Verified Commands

- **Typecheck:** Not applicable (Python — no type checker configured)
- **Lint:** Not applicable (no linter configured)
- **Test:** Not applicable (no test suite present)

### Scoped Workflow

No scoped verification commands available — this repo has no type checker, linter, or test framework configured.

### Validation Summary

- **Result:** partial
- **Runtime:** Python 3.13.13 — available
- **Libraries:** pandas 2.2.3, numpy 2.3.5, scipy 1.17.1, scikit-learn 1.6.1 — all installed and importable
- **Script execution:** FAIL — `TypeError: Population must be a sequence` at `random.sample(X.index, ...)` (line 28). The code uses APIs removed in modern pandas (`.ix` indexer, `Index` as a sequence for `random.sample`).
- **Blocker:** `runtime_unavailable` — script incompatible with installed library versions. Code written for pandas ~0.17; modern pandas (2.x) removed `.ix` and changed `Index` behavior.

## Sandbox Snapshot

- **Snapshot ID:** `i1l2ligek27i6xm0qul6i`
- **Captured:** 2026-08-07T19:06:28Z
- **Dev stack healthy:** yes (runtime + libraries available; script has compatibility issues with modern pandas)

## Bibliography

Product Atlas nodes already exist for this repo with comprehensive coverage:

- **Product:** `nx2-cross-validation-model-comparison` (bib_1sVv4dM8)
  - **Concept:** `classifier-evaluation-comparison` (bib_rFwNNR2T)
    - Feature: `nx2-fold-cross-validation`
    - Feature: `class-undersampling`
    - Feature: `model-ranking-by-roc-auc`
    - Feature: `statistical-significance-testing`
    - Feature: `example-demonstration`

File source `Nx2cv.py` attached to product node during this scan. Total: 1 source attached.

## Security Scan

> **Note:** security_scan_not_triggered — `trigger_security_onboarding` tool not available in this worker's toolset. Trigger manually if needed.

## Runbooks

Populated by autobuild-runbooks skill when requested. See `.obvious/runbooks/` after that skill runs.

