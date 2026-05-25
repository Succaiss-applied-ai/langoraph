# CI and Automation

`langoraph` uses a scoped CI graph inspired by OpenClaw's release discipline, scaled down for a Go library.

## CI

The main CI workflow runs on `main`, pull requests, and manual dispatches.

- `preflight` classifies the diff and skips Go lanes for docs-only changes.
- `workflow-sanity` parses workflow YAML when automation files change.
- `format`, `vet`, `tests`, and `race` run the local Go gate.
- `docs` checks Markdown links when docs or workflow text changes.
- `ci-result` is the single aggregate status to make branch protection simpler.
- `PR Hygiene` requires real behavior proof for external PRs, with `proof: required` and `proof: override` labels for maintainer control.

Manual dispatch with `full_suite=true` forces every lane.

## Security

The security workflow runs:

- `govulncheck`
- CodeQL for Go
- OSSF Scorecard on trusted refs

## Codex

The Codex workflow supports two modes:

- Automatic read-only review for same-repository PRs when `OPENAI_API_KEY` is configured.
- Maintainer-triggered review with `/codex review` or `@codex review` comments.

It runs through `openai/codex-action@v1` with a read-only sandbox and posts the final Codex response back to the PR.

## Releases

Pushing a `v*.*.*` tag validates formatting, vet, and race tests, then creates or updates a GitHub release with notes from commits since the previous tag.
