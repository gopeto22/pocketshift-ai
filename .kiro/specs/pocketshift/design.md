# PocketShift AI — Design

## Architecture
```
Streamlit (app.py)
  ├── Screen A Command Centre
  ├── Screen B Play Forensics      ──┐
  ├── Screen C Counterfactual Lab    │  all read cached processed tables
  ├── Screen D Ask PocketShift       │
  └── Screen E Player Preparation  ──┘
         │
src/ (pure Python, Streamlit-agnostic, unit-testable)
  config.py            paths, constants, thresholds, evidence types
  ingestion.py         CSV load, NA handling, strict joins, Parquet build
  validation.py        join integrity, key uniqueness, NA safety checks
  features.py          pressure labels, pre-snap feature frame (leakage-guarded)
  matchup_graph.py     blocker–rusher interaction network (NetworkX)
  scenarios.py         similarity search, empirical rates, bootstrap CIs, support
  modelling.py         grouped-CV pressure model (logreg / HGB), calibration
  retrieval.py         DuckDB structured queries + TF-IDF play-text search
  evidence.py          evidence objects + deterministic quality reviewer
  agents.py            tool registry + orchestration (Bedrock or deterministic)
  bedrock_client.py    Converse API tool-use loop + graceful degradation
  performance_adapter.py  PerformanceContextAdapter + preparation brief builder
  visualizations.py    Plotly charts, graph figure, theme
```

## Data model (processed Parquet)
- `plays_enriched.parquet` — one row/play: situation + personnel + derived
  pressure_event, sack, n_rushers, n_blockers, qb name, pre-snap feature columns.
- `matchups.parquet` — one row per blocker→rusher assignment with outcomes.
- `player_profiles.parquet` — per nflId: position, exposure counts, outcomes.
- `team_patterns.parquet` — per team: pressure rates, common configs.

## Pressure label
pressure_event = max over Pass Rush rows of (hit|hurry|sack). Documented proxy.

## Counterfactual method
Matched comparison: filter real plays on controllable context (down bucket,
distance bucket, offense/defense optional, personnel, formation, box, coverage
when retrospective). Compare pressure rate of context WITH vs WITHOUT the
modification (e.g. extra blocker: n_blockers>=6, or RB/TE chip present).
Report both rates, difference, n per arm, and bootstrap 95% CI on the difference.
Below MIN_SUPPORT per arm → unsupported warning. Never causal language.

## Model
Target: pressure_event. Features: down, yardsToGo, quarter, personnelO/D,
offenseFormation, defendersInBox, absoluteYardlineNumber, score diff, play-action.
Split: GroupKFold by gameId (no same-game leakage). Models: calibrated logistic
regression + HistGradientBoosting. Report ROC-AUC, Brier, calibration, n, prevalence
vs base-rate baseline. Leakage guard: assert no post-snap column enters X.

## Agent tools (real functions in agents.py, wrapping src modules)
lookup_play, get_play_participants, get_pass_protection_graph, get_player_history,
get_pressure_events, find_similar_plays, calculate_empirical_pressure_rate,
compare_protection_configurations, evaluate_protection_scenario,
estimate_historical_alternative, get_position_demands, summarize_observed_exposure,
build_preparation_brief. Each returns a structured dict with sources.

## Bedrock loop
Converse API with toolConfig. Loop: send → read toolUse → validate args →
execute real function → return toolResult → repeat until end_turn. Capture a
sanitised trace. On any boto/credential/model error → deterministic fallback,
flag "AI agent unavailable". Model id from env, else auto-pick accessible Claude.

## Evidence reviewer (deterministic, authoritative)
Checks: cited (gameId, playId) exist; no injury/medical/fatigue claim strings;
tool-call trace entries all map to executed functions; empirical claims carry n.
Any failure → answer blocked / downgraded. LLM cannot override.

## Theme
Dark navy/charcoal, cyan/green accent, amber caution, red breakdown. Plotly dark
template, labelled axes/units, compact cards.
