"""Dependency-free ensemble AI-text detector.

Combines several calibrated stylometric signals into a single AI-likelihood
score. This is a heuristic — not an ML classifier — but far more robust than
a single-feature detector: it fuses burstiness, vocabulary richness,
contraction rate, formal-connector density, AI-cliche phrases, sentence-opening
diversity and punctuation variety, and reports a confidence based on length.
"""

from __future__ import annotations

import math
import re

# Formal connectives over-represented in LLM prose.
CONNECTORS = {
    "however", "moreover", "furthermore", "additionally", "consequently",
    "therefore", "thus", "hence", "nonetheless", "nevertheless", "accordingly",
    "subsequently", "overall", "ultimately", "notably", "importantly",
    "specifically", "particularly", "essentially", "fundamentally",
}

# Multi-word phrases strongly associated with LLM output.
CLICHES = [
    "it is important to note", "it is worth noting", "plays a crucial role",
    "plays a vital role", "plays a significant role", "in today's fast-paced",
    "in the realm of", "in the world of", "delve into", "delving into",
    "navigate the", "navigating the", "a testament to", "underscore",
    "underscores", "at the end of the day", "when it comes to",
    "a wide range of", "a variety of", "seamless", "seamlessly",
    "cutting-edge", "game-changer", "game changer", "revolutionize",
    "harness the power", "unlock the potential", "the ever-evolving",
    "ever-evolving", "rich tapestry", "tapestry of", "shed light on",
    "pave the way", "paving the way", "in conclusion", "to sum up",
    "first and foremost", "last but not least", "on the other hand",
]

CONTRACTION_RE = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z']+")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _mattr(words: list[str], window: int = 40) -> float:
    """Moving-average type-token ratio — length-robust vocabulary richness."""
    lw = [w.lower() for w in words]
    if len(lw) <= window:
        return len(set(lw)) / len(lw) if lw else 0.0
    ratios = []
    for i in range(len(lw) - window + 1):
        chunk = lw[i:i + window]
        ratios.append(len(set(chunk)) / window)
    return sum(ratios) / len(ratios)


def analyze(text: str) -> dict:
    text = text.strip()
    words = WORD_RE.findall(text)
    sents = _sentences(text)
    n_words = len(words)
    n_sents = len(sents)

    # --- raw features -------------------------------------------------------
    lengths = [len(WORD_RE.findall(s)) for s in sents] or [0]
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths) if lengths else 0.0
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_len if mean_len > 0 else 0.0  # burstiness

    mattr = _mattr(words)
    contraction_rate = len(CONTRACTION_RE.findall(text)) / n_words * 100 if n_words else 0.0

    lw = [w.lower() for w in words]
    connector_hits = sum(1 for w in lw if w in CONNECTORS)
    connector_density = connector_hits / n_sents if n_sents else 0.0

    low = text.lower()
    cliche_hits = sum(low.count(p) for p in CLICHES)

    first_words = [WORD_RE.findall(s)[0].lower() for s in sents if WORD_RE.findall(s)]
    opening_diversity = len(set(first_words)) / len(first_words) if first_words else 1.0

    expressive = sum(1 for ch in ("!", "?", ";", "—", "–", "…", "(", '"', "'") if ch in text)
    # repeated bigrams
    bigrams = list(zip(lw, lw[1:]))
    bigram_rep = 1 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0

    # --- map each feature to an AI-likelihood contribution (0..1) -----------
    ai_burst = _clamp((0.55 - cv) / 0.45)                       # uniform lengths -> AI
    ai_vocab = _clamp((0.82 - mattr) / 0.30)                    # low richness -> AI
    ai_contr = _clamp((1.6 - contraction_rate) / 1.6)          # few contractions -> AI
    ai_conn = _clamp((connector_density - 0.10) / 0.45)        # many connectors -> AI
    ai_cliche = _clamp(cliche_hits / 3.0)                      # 3+ cliches -> strong AI
    ai_open = _clamp((0.85 - opening_diversity) / 0.55) if n_sents >= 4 else 0.0
    ai_punct = _clamp((3 - expressive) / 3.0)                  # flat punctuation -> AI
    ai_band = _clamp(1 - abs(mean_len - 21) / 12) * 0.8        # 15-27 wpm band -> AI-ish
    ai_rep = _clamp((bigram_rep - 0.05) / 0.25)

    weighted = [
        (ai_burst, 0.20, "sentence-length uniformity"),
        (ai_vocab, 0.13, "vocabulary richness"),
        (ai_contr, 0.12, "contraction rate"),
        (ai_conn, 0.14, "formal connector density"),
        (ai_cliche, 0.13, "AI cliche phrases"),
        (ai_open, 0.10, "sentence-opening diversity"),
        (ai_punct, 0.07, "punctuation variety"),
        (ai_band, 0.06, "sentence-length band"),
        (ai_rep, 0.05, "phrase repetition"),
    ]
    raw = sum(v * w for v, w, _ in weighted)
    # gentle sigmoid to spread scores around the 0.5 midpoint
    score = _clamp(1 / (1 + math.exp(-(raw - 0.45) * 6)))

    confidence = _clamp(n_words / 120)  # short text -> low confidence

    if score >= 0.60:
        label, verdict = "ai", "Likely AI-generated"
    elif score <= 0.40:
        label, verdict = "human", "Likely human-written"
    else:
        label, verdict = "mixed", "Uncertain / mixed signals"
    if confidence < 0.35:
        label, verdict = "mixed", "Too little text for a confident verdict"

    contributors = sorted(
        ({"signal": name, "weight": w, "ai_score": round(v, 3)} for v, w, name in weighted),
        key=lambda c: c["ai_score"] * c["weight"],
        reverse=True,
    )[:4]

    return {
        "ai_likelihood": round(score, 4),
        "percent": round(score * 100),
        "label": label,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "signals": {
            "words": n_words,
            "sentences": n_sents,
            "avg_sentence_len": round(mean_len, 1),
            "sentence_len_variation": round(cv, 3),
            "vocabulary_richness": round(mattr, 3),
            "contraction_rate": round(contraction_rate, 2),
            "connector_density": round(connector_density, 3),
            "cliche_hits": cliche_hits,
            "opening_diversity": round(opening_diversity, 3),
        },
        "top_signals": contributors,
    }
