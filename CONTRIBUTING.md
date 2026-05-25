# Contributing

Thanks for improving `langoraph`. Keep changes small, tested, and API-conscious.

## Local Gate

```bash
make ci
```

Without `make`:

```bash
gofmt -w $(git ls-files '*.go')
go vet ./...
go test ./...
go test -race ./...
```

## Pull Requests

External PRs must include real behavior proof in the template. CI checks for these sections so reviewers can see what changed, where it was tested, and what risk remains. Maintainers can force the gate with `proof: required` or bypass it with `proof: override`.

Maintainers can ask Codex for a review by commenting:

```text
/codex review
```

or:

```text
@codex review
```

The Codex workflow requires an `OPENAI_API_KEY` repository secret. Optional repository variables:

- `CODEX_MODEL`
- `CODEX_EFFORT`

## Compatibility

The module currently targets the Go version in `go.mod`. Do not raise that version casually; it changes downstream installation expectations.
