"""Retrieval layer: DuckDB structured queries + TF-IDF play-text search.

Keeps everything local and deterministic. The agent uses this to ground answers
in actual rows (RAG over play descriptions + structured aggregates).
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass
class PlayHit:
    game_id: int
    play_id: int
    score: float
    description: str


class RetrievalIndex:
    """TF-IDF index over play descriptions + a DuckDB view of enriched plays."""

    def __init__(self, plays_enriched: pd.DataFrame):
        self.plays = plays_enriched.reset_index(drop=True).copy()
        self.plays["playDescription"] = self.plays["playDescription"].fillna("").astype(str)
        self._vec = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
        self._matrix = self._vec.fit_transform(self.plays["playDescription"])
        self._con = duckdb.connect(database=":memory:")
        self._con.register("plays", self.plays)

    # --- textual search ---
    def search(self, query: str, limit: int = 5) -> list[PlayHit]:
        if not query.strip():
            return []
        q = self._vec.transform([query])
        sims = linear_kernel(q, self._matrix).ravel()
        top = np.argsort(sims)[::-1][:limit]
        hits = []
        for i in top:
            if sims[i] <= 0:
                continue
            r = self.plays.iloc[i]
            hits.append(
                PlayHit(
                    game_id=int(r["gameId"]),
                    play_id=int(r["playId"]),
                    score=float(sims[i]),
                    description=str(r["playDescription"])[:220],
                )
            )
        return hits

    # --- structured SQL ---
    def sql(self, query: str) -> pd.DataFrame:
        return self._con.execute(query).df()

    def pressure_rate_by(self, column: str) -> pd.DataFrame:
        safe = {
            "possessionTeam",
            "defensiveTeam",
            "offenseFormation",
            "down_bucket",
            "distance_bucket",
            "pff_passCoverage",
        }
        if column not in safe:
            raise ValueError(f"column not allowed: {column}")
        return self.sql(
            f"""
            SELECT "{column}" AS group_value,
                   COUNT(*) AS n_plays,
                   AVG(pressure_event) AS pressure_rate,
                   AVG(sack_event) AS sack_rate
            FROM plays
            WHERE "{column}" IS NOT NULL
            GROUP BY "{column}"
            HAVING COUNT(*) >= 20
            ORDER BY pressure_rate DESC
            """
        )

    def high_pressure_plays(self, limit: int = 10) -> pd.DataFrame:
        return self.sql(
            f"""
            SELECT gameId, playId, possessionTeam, defensiveTeam,
                   playDescription, hits, hurries, sacks, n_rushers, n_blockers
            FROM plays
            WHERE sack_event = 1
            ORDER BY (hits + hurries + sacks) DESC
            LIMIT {int(limit)}
            """
        )
