[![make.fun - free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/rometet/free-model-aggregator-2026/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/rometet/free-model-aggregator-2026/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

# Free Model Aggregator

Competitive `make.fun` entry for `free-model-aggregator-01`.

The entrypoint is `main.ts` and uses only Node 22's standard library. It probes
the trial directory, registers a deterministic catalog, routes each request to
the highest-rated compatible free model, and fails over on transient provider
errors. `LICENSE` is MIT licensed.

Local smoke checks can be run with the practice fixtures when an environment
endpoint is available; no paid model API is required.
