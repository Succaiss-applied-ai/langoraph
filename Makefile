.PHONY: check ci coverage fmt fmt-check test test-race tidy tidy-check vet

GO ?= go
GOFILES := $(shell git ls-files '*.go')
CHANGED_GOFILES := $(shell git diff --name-only --diff-filter=ACMR origin/main...HEAD -- '*.go' 2>/dev/null)

fmt:
	gofmt -w $(GOFILES)

fmt-check:
	@if [ -z "$(CHANGED_GOFILES)" ]; then \
		echo "No Go files changed."; \
	else \
		test -z "$$(gofmt -l $(CHANGED_GOFILES))" || (gofmt -l $(CHANGED_GOFILES); exit 1); \
	fi

tidy:
	$(GO) mod tidy

tidy-check:
	$(GO) mod tidy
	git diff --exit-code -- go.mod go.sum

vet:
	$(GO) vet ./...

test:
	$(GO) test ./...

test-race:
	$(GO) test -race ./...

coverage:
	$(GO) test -covermode=atomic -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out

check: fmt-check tidy-check vet test

ci: check test-race
