# Free Model Aggregator

Competitive `make.fun` entry for `free-model-aggregator-01`.

The entrypoint is `main.ts` and uses only Node 22's standard library. It probes
the trial directory, registers a deterministic catalog, routes each request to
the highest-rated compatible free model, and fails over on transient provider
errors. `LICENSE` is MIT licensed.

Local smoke checks can be run with the practice fixtures when an environment
endpoint is available; no paid model API is required.
