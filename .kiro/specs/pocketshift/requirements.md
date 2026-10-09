# PocketShift AI — Requirements

## Problem
NFL analytics describe what happened on a pass-protection play. Coaches still
spend scarce time working out what to investigate changing, what evidence
supports an alternative, and how to turn that into preparation. PocketShift
closes that gap with evidence-typed counterfactual investigation.

## Users
- Offensive line / pass-protection coach (primary).
- Performance & conditioning staff (secondary, via preparation brief).
- Hackathon judges evaluating a 3-minute live demo.

## Functional requirements (EARS-style)

### R1 Data ingestion & validation
- WHEN the app starts, THE SYSTEM SHALL load games, plays, players and PFF
  scouting CSVs with documented NA handling and correct dtypes.
- THE SYSTEM SHALL join strictly on composite keys (never playId alone).
- THE SYSTEM SHALL build enriched play, matchup, player-profile and team-pattern
  tables and persist them as Parquet under data/processed/.
- IF a required CSV is missing, THE SYSTEM SHALL fail with a clear message, not
  silently fabricate data.

### R2 Pressure labelling
- THE SYSTEM SHALL label a play as a pressure event when any Pass Rush row has
  pff_hit, pff_hurry or pff_sack == 1, and SHALL document it as a scouting proxy.

### R3 Play forensics
- WHEN a user selects a play, THE SYSTEM SHALL show observed situation, personnel,
  participants, recorded blocker–rusher assignments and pressure/sack events.
- THE SYSTEM SHALL render an interactive blocker–rusher interaction graph.

### R4 Historical comparison & counterfactual
- WHEN a user requests an alternative, THE SYSTEM SHALL retrieve comparable plays
  on controllable context and compute an empirical rate WITH sample size.
- IF matched support is below threshold, THE SYSTEM SHALL return an explicit
  unsupported-scenario warning instead of a number.
- THE SYSTEM SHALL label every estimate observed | historical | model | hypothetical.

### R5 AI agent
- WHEN Bedrock is available, THE SYSTEM SHALL run a real tool-use loop: the model
  selects tools, the system executes real analytical functions, and the final
  answer cites valid game/play source identifiers.
- WHEN Bedrock is unavailable, THE SYSTEM SHALL degrade gracefully to deterministic
  answers and clearly indicate the AI agent is unavailable.

### R6 Preparation brief
- WHEN a user generates a brief, THE SYSTEM SHALL produce film-study priority,
  technical focus, workload questions (proxy-labelled), hydration/recovery
  checklist and an explicit "practitioner review required" gate, using only
  dataset-derived facts plus optional fictional context.

### R7 Evidence & quality review
- THE SYSTEM SHALL validate every AI answer deterministically (valid sources, no
  medical/injury claims, no fabricated tool calls) and SHALL block display when
  validation fails.

## Non-functional
- Runs without Bedrock. Featured demo loads quickly (cached joins).
- No secrets in code/logs. Raw dataset not redistributed.
- Tests run and pass.

## Acceptance (MVP definition of done)
App starts; four CSVs load & validate; featured play retrieved accurately;
blocker–rusher graph displayed; one supported alternative explorable; empirical
calcs reproducible; no unsupported causal claims; one real Bedrock tool call when
AWS available; AI answers carry valid references; preparation brief generates;
tests run; README with run instructions; 3-minute demo completable.
