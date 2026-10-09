# PocketShift AI — Tasks

## P0 Data & working dashboard
- [x] 1. Scaffold project, steering, specs, env, deps.
- [x] 2. config.py — paths, constants, thresholds, evidence types.
- [x] 3. ingestion.py — load CSVs, NA handling, strict joins, Parquet build.
- [x] 4. validation.py — join integrity + key/NA checks.
- [x] 5. features.py — pressure labels + leakage-guarded pre-snap frame.
- [x] 6. matchup_graph.py — blocker–rusher interaction network.
- [x] 7. visualizations.py — theme + Plotly figures + graph.
- [x] 8. app.py — Command Centre + Play Forensics (featured play works).
- [x] 9. profile_data.py script → outputs/data_profile.md + Parquet.

## P1 Historical scenario analysis
- [x] 10. scenarios.py — similarity search, empirical rates, bootstrap CIs, support.
- [x] 11. Counterfactual Lab screen with what-if controls + honest warnings.

## P2 Genuine Bedrock agent
- [x] 12. retrieval.py — DuckDB queries + TF-IDF play-text search.
- [x] 13. evidence.py — evidence objects + deterministic reviewer.
- [x] 14. modelling.py — grouped-CV pressure model + calibration report.
- [x] 15. bedrock_client.py — Converse tool-use loop + graceful degradation.
- [x] 16. agents.py — tool registry + orchestration + trace.
- [x] 17. Ask PocketShift screen with visible execution trace + evidence panel.

## P3 NutriElite-inspired preparation
- [x] 18. performance_adapter.py — adapter + brief builder.
- [x] 19. Player Preparation screen + Create Preparation Brief + Markdown export.

## P4 Polish & package
- [x] 20. Tests (data joins, pressure labels, scenarios, evidence, agents, leakage).
- [x] 21. README, demo script, screenshots, final run verification.

## P5 Live Bedrock + Energy/Fatigue side-quest
- [x] 22. Fix Bedrock region (eu-central-1) + model candidates; confirm live tool loop.
- [x] 23. Harden tool results to JSON-safe types for Converse.
- [x] 24. tracking.py — snap-gated Energy Expenditure Index + fatigue signal.
- [x] 25. Energy & Fatigue screen (curve, per-play energy, fatigue↔outcome).
- [x] 26. Agent tools get_player_fatigue / get_play_energy wired + used live.
- [x] 27. Tracking tests + opt-in live-Bedrock test; README/DEMO updates.
