# PocketShift AI — Demo scripts

## One-minute pitch

"PocketShift AI turns an NFL pass-protection failure into a coaching decision.
It reconstructs who blocked whom, retrieves comparable historical plays, and
estimates a protection alternative with real sample sizes and honest uncertainty —
never a made-up number. A live Amazon Bedrock agent (Claude Sonnet 4.5) chooses and
runs real analytical tools — including a tracking-derived energy and fatigue metric
that flags when players lose explosive output late in games — cites the exact plays
behind every claim, and a deterministic reviewer blocks anything unsupported. Then
it hands the lesson to the performance staff as a review-gated preparation brief.
From pressure, to preparation."

## Three-minute live demo

Prep: `streamlit run app.py`, land on **Command Centre**.

### 0:00–0:25 — The problem
- Point to the featured case card: Game 2021090900 · Play 2279, Dak Prescott
  sacked by Shaquil Barrett.
- "Every play is a mountain of data, but it still takes staff time to see what
  happened and decide what to change." Click **Investigate this play**, open
  **Play Forensics**.

### 0:25–1:00 — Forensics
- Show the situation cards (Q2, 2 & 10, Shotgun, Quarters) and outcome (pressure
  YES, sack YES, 4 rushers vs 5 blockers).
- Walk the **blocker → rusher interaction graph**: Tyron Smith (LT) assigned to
  Barrett (ROLB), the sacker shows in the pass-rusher table with pff_sack = 1.
- "PocketShift reconstructs the observed matchups from PFF assignments — not a
  fabricated spatial view — and shows exactly where to investigate."

### 1:00–1:30 — Counterfactual Lab
- Switch to **Counterfactual Lab**, scenario "keep an extra blocker in", match on
  down + distance + formation, estimate.
- Show the bar chart + metrics: with extra protection 42.5% (n=228) vs standard
  33.9% (n=888), difference with a 95% CI.
- "Instead of claiming an alternative outcome, PocketShift shows what comparable
  history supports — and flags that this looks like selection bias, not a causal
  effect. Honesty over hype."

### 1:30–2:05 — Agent (live Bedrock)
- Go to **Ask PocketShift**, click "Why did this protection fail?"
- Sidebar shows **AI agent online · eu.anthropic.claude-sonnet-4-5-20250929-v1:0**.
- Point at the **execution trace** — the actual tool calls Claude chose, including
  `get_play_energy` — then the answer: it diagnoses a *coverage sack* (5-on-4, no
  breakdowns, Barrett double-teamed) and cites the tracking energy figures.
- "A real tool-using Bedrock agent. It chose the tools, pulled live tracking
  energy data, and the deterministic reviewer only lets it through if the sources
  are valid and nothing is overclaimed." (Falls back to deterministic mode, clearly
  labelled, if Bedrock is unavailable — same real tools either way.)

### 2:05–2:30 — Energy & Fatigue (side-quest)
- Go to **Energy & Fatigue**. Tab 1: pick a high-load player and show the energy
  load bars rising while the peak-acceleration trend line falls — the fatigue
  signature. Tab 3: the pressure rate climbing across energy-load tertiles.
- "From frame-by-frame tracking: who's expending the most, who loses explosive
  output late, and how effort tracks with pressure. This is the refuelling and
  rotation signal a performance coach asked for — a proxy, never a medical claim."

### 2:30–2:50 — Preparation bridge
- Go to **Player Preparation**, pick Tyron Smith, click **Create Preparation
  Brief**.
- Show the game finding, technical priority, workload questions (labelled a
  proxy), and the bold **Human review required** gate. Download the Markdown.
- "The same intelligence connects tactics to how teams prepare individual
  athletes — with staff approval, and no medical or injury claims."

### 2:50–3:00 — Close
- "PocketShift takes NFL data beyond what happened — into what to investigate,
  what the evidence supports, and how to turn findings into human decisions."
