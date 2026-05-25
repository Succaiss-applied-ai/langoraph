# Security Policy

Please report vulnerabilities with GitHub private vulnerability reporting:

https://github.com/Succaiss-applied-ai/langoraph/security/advisories/new

Do not open public issues for credential leaks, provider request signing bugs, prompt injection paths that expose secrets, or denial-of-service issues.

## Supported Version

The `main` branch is the supported development line until tagged releases are established.

## Scope

Security-sensitive areas include:

- LLM provider request construction and headers.
- Streaming response parsing.
- Timeout and retry behavior.
- Any future code that reads local files, environment variables, or executes tools.
