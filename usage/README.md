# Provider token usage logs

Every physical HTTP attempt emits `model.request.started` and, when its response is read, closed or fails, one `model.token_usage` JSON record. Filter on the latter event and deduplicate by `call_id`. Do not sum start records, legacy logical-response counters, cache subsets, reasoning subsets or multiple streaming snapshots. Retries, schema repair and provider fallback each receive a distinct call ID.

Counts come only from the provider response. Missing fields are `null`; reported zero remains zero. `usage_status` is `reported`, `partial`, `unknown` or `invalid`. Interrupted SSE without a final marker is partial even if an intermediate usage frame exists. Preserve unknown attempts in coverage reports instead of claiming their total is exact. `provider_usage` retains numeric details, including cache, reasoning and modality breakdowns. It never includes credentials, prompts, completions or error bodies.

`usage.WithScope` adds task, trace, logical parent-call, run, thread or document IDs without changing provider request behavior. `usage.Wrap` copies the HTTP client and preserves its underlying connection pool, deadlines and redirects. This is a usage receipt, not reconciliation against a supplier invoice. A process crash can leave a start without a terminal record; classify it as unknown.
