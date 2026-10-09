# Analytics Integrity — PocketShift AI

## Evidence typing (every number must declare its type)
- `observed` — read directly from the dataset for a specific play.
- `historical` — empirical rate over a matched subset of real plays (show n).
- `model` — output of a fitted statistical model (show metric + uncertainty).
- `hypothetical` — not estimable with current data; say so explicitly.

## Hard rules
1. Every historical comparison displays its sample size (n).
2. Empty / tiny comparison groups never produce a percentage. Below the minimum
   support threshold, return: "Not enough comparable historical evidence to
   estimate this scenario reliably."
3. Never call an observational comparison a proven causal effect. Blocking
   choices are not randomised.
4. No fabricated tracking geometry. The matchup graph is an interaction network,
   not a spatial reconstruction, unless real tracking x/y is actually loaded.
5. Workload = observed play involvement count. Label it a proxy. Never infer
   injury risk, fatigue, or any physiological/medical state from scouting data.
6. The evidence reviewer (deterministic) can veto any answer. An LLM answer can
   never override a deterministic validation failure.
7. AI tool-execution traces must correspond to functions that actually ran.
   Never display a fabricated tool call.

## Uncertainty
Prefer bootstrap confidence intervals for empirical rates where n allows.
Report ROC-AUC, Brier score, calibration, held-out n, and class prevalence for
models. If a model is weak, report it; do not conceal.
