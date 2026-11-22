## Checks

`make check` runs lint, format check and tests (the same steps as `.github/workflows/ci.yml`). Pre-commit hooks (`pre-commit install`) run ruff on staged files.

## Agent skills

### Issue tracker

GitHub Issues on `gr8monk3ys/llm-client`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, label string equal to role name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
