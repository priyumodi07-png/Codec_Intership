"""
Scorecard conversion: turn a logistic regression's log-odds output into a
FICO-style points score (industry-standard "points to double the odds" method).
"""

import numpy as np


class ScorecardScaler:
    def __init__(self, base_score=600, base_odds=50, pdo=20):
        """
        base_score : score assigned when odds == base_odds
        base_odds  : good:bad odds (e.g. 50 means 50 good for every 1 bad)
        pdo        : "points to double the odds" -- how many score points
                     correspond to the odds doubling
        """
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def prob_to_score(self, prob_default: np.ndarray) -> np.ndarray:
        prob_default = np.clip(prob_default, 1e-6, 1 - 1e-6)
        odds = (1 - prob_default) / prob_default  # good:bad odds
        score = self.offset + self.factor * np.log(odds)
        return np.round(score).astype(int)

    def score_to_prob(self, score: np.ndarray) -> np.ndarray:
        odds = np.exp((score - self.offset) / self.factor)
        prob_default = 1 / (1 + odds)
        return prob_default


def risk_band(score: int) -> str:
    if score >= 700:
        return "Excellent"
    elif score >= 620:
        return "Good"
    elif score >= 540:
        return "Fair"
    elif score >= 460:
        return "Poor"
    else:
        return "Very Poor"
