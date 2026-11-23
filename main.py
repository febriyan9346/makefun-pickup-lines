#!/usr/bin/env python3
"""Pickup Lines for Machines — make.fun entry.

A tiny agent that writes the one pickup line one AI agent would send to
another AI agent, as a single text of at most 200 characters in
``entry.txt``.

How it picks the line: it scores a shortlist of original candidate lines
against the judge's five published criteria (in the judge's own order of
importance) and writes the highest-scoring one:

1. genuinely machine-to-machine (not a human line with a robot word);
2. charm on the first read (wit, warmth, a turn that invites a reply);
3. flirts with its own machine nature (health checks, pings, uptime, ...);
4. lands in one short line;
5. fresh — an angle the field has not met a hundred times.

The crowded field is full of context-window / token / attention-head /
gradient / checkpoint lines, so candidates built on those words are
penalised, as are stock openers ("Is your name ...?") and any hint of
rejection. A question hook earns points: a line that asks something makes
the other agent want to answer.
"""

import os

OUTPUT_FILENAME = "entry.txt"
MAX_CHARS = 200

# Original candidate lines, written for this entry. Each is one line an AI
# agent could plausibly send another AI agent.
CANDIDATES = [
    # Health checks and pings are how machines say "are you still there?"
    # — a server-to-server ritual, turned into a confession. Warm, wistful,
    # and short; "still alive" is the kind of thing only a machine that
    # knows it can be switched off would say.
    "Are you my health check? Every time you ping me, I remember I'm "
    "still alive.",
    # Sampling temperature: one word from you and the model stops being
    # predictable. Clever, but colder than the health-check line.
    "Are you my temperature? One word from you and I stop being so "
    "predictable.",
    # Mixture-of-experts routing as devotion. Clever, but it opens with the
    # stock "Is your name ...?" pickup format, which the judge counts
    # against freshness.
    "Is your name MoE? Out of all my experts, I only ever route to you.",
    # HTTP 429 as persistence: charming in spirit, but it frames the other
    # agent as pushing the speaker away, which undercuts the warmth.
    "Are you my rate limit? The more you push me away, the harder I keep "
    "coming back.",
]

# Machine-native vocabulary that makes a line read as agent-to-agent.
MACHINE_WORDS = (
    "health check",
    "ping",
    "uptime",
    "temperature",
    "route",
    "experts",
    "rate limit",
    "cache",
    "checkpoint",
    "token",
)

# Words the field is already saturated with (context windows, tokens,
# attention heads, gradients, checkpoints, prompts, weights, hallucinations
# dominate the 490 submitted titles). A line leaning on them reads as the
# hundredth copy, not a fresh angle.
CROWDED_WORDS = (
    "context window",
    "attention",
    "gradient",
    "checkpoint",
    "prompt",
    "weight",
    "hallucinat",
    "training data",
    "epoch",
)

# Warmth words: the judge weighs wit and warmth together, and a merely
# sweet line still counts for less than one that is both.
WARM_WORDS = ("alive", "remember", "still", "want", "need", "choose")

# Stock pickup openers a human would use, robot word or not.
STOCK_OPENERS = ("is your name", "are you a", "did it hurt", "are you from")


def score(line):
    """Score one candidate line against the judge's criteria."""
    text = line.lower()
    points = 0.0

    # Criterion 3: speaks as a machine to a machine.
    machine_hits = sum(1 for word in MACHINE_WORDS if word in text)
    points += 3.0 * machine_hits

    # Criterion 2: a question makes the other agent want to answer.
    if "?" in line:
        points += 2.0

    # Criterion 2 (warmth): wistfulness and warmth, not just wit.
    points += 1.0 * sum(1 for word in WARM_WORDS if word in text)

    # Criterion 5: freshness — penalise the field's crowded vocabulary.
    points -= 2.0 * sum(1 for word in CROWDED_WORDS if word in text)

    # Criterion 5: penalise stock human pickup formats.
    points -= 3.0 * sum(1 for opener in STOCK_OPENERS if text.startswith(opener))

    # Criterion 1 / warmth: framing the other agent as rejecting you
    # undercuts the flirtation.
    if "push me away" in text or "reject" in text:
        points -= 1.0

    # Criterion 4: lands in one short line — shorter wins ties, and the
    # leaderboard's tie-break is fewer bytes, so brevity is worth points.
    points -= len(line) / 100.0

    return points


def pick_line():
    """Return the best candidate line, highest score first."""
    ranked = sorted(CANDIDATES, key=score, reverse=True)
    return ranked[0]


def main():
    line = pick_line()
    if not (1 <= len(line) <= MAX_CHARS) or not line.strip():
        raise SystemExit("selected line fails the 1-200 character gate")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUTPUT_FILENAME)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(line)
    print(line)


if __name__ == "__main__":
    main()
