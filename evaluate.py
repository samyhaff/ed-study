import argparse
import itertools
from collections import Counter

import numpy as np
import scipy.stats as st

from ed import parse_answer, ranked, read_jsonl

CATEGORIES = ["LOW_OVERLAP", "MULTIPLE_CATEGORIES", "AMBIGUOUS_SUBSTRING", "HIGH_OVERLAP"]


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (values.mean() if len(values) else 0.0), 0.0
    low, _ = st.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=st.sem(values))
    return values.mean(), values.mean() - low


def flatten(records):
    rows = []
    for r in records:
        for i, gold in enumerate(r["ground_truths"]):
            meta = r["metadata"] | {"mentions": r["metadata"]["mentions"][i : i + 1]} if r["metadata"] else {}
            if "cleaned_candidates" in meta:
                meta["cleaned_candidates"] = meta["cleaned_candidates"][i]
            rows.append({"ground_truth": gold.replace("_", " "), "prediction": r["predictions"][i], "candidates": r["candidates"][i], "metadata": meta})
    return rows


def in_candidates(r, key="candidates"):
    return r["ground_truth"] in [c[0] for c in r[key]]


def samples(rows):
    for r in rows:
        candidates = r["metadata"].get("cleaned_candidates", r["candidates"])
        outputs = r["metadata"].get("mentions", [{}])[0].get("model_outputs", [])
        if outputs and r["ground_truth"] in [c[0] for c in candidates]:
            answers = [parse_answer(o) for o in outputs]
            yield r["ground_truth"], [candidates[a - 1][0] if 0 < a <= len(candidates) else None for a in answers]


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def maj_at_k(gold, predictions, k):
    score, subsets = 0.0, 0
    for subset in itertools.combinations(predictions, k):
        subsets += 1
        votes = Counter(p for p in subset if p is not None)
        if votes:
            top = [p for p, v in votes.items() if v == max(votes.values())]
            score += (gold in top) / len(top)
    return score / subsets


parser = argparse.ArgumentParser()
parser.add_argument("benchmark", choices=["zelda", "zeshel", "gladis"])
parser.add_argument("predictions")
parser.add_argument("--exclude_train_data")
parser.add_argument("--top_k", type=int)
parser.add_argument("--sc", action="store_true")
args = parser.parse_args()

rows = read_jsonl(args.predictions)
if args.benchmark == "zelda":
    rows = flatten(rows)
if args.benchmark == "gladis":
    rows = [r | {"ground_truth": r["long_term"]} for r in rows]
if args.exclude_train_data:
    seen = {r["mention_id"] for r in read_jsonl(args.exclude_train_data) if "mention_id" in r}
    rows = [r for r in rows if r["mention_id"] not in seen]

correct = lambda r: float(r["prediction"] == r["ground_truth"])
print(f"n={len(rows)}")
print("accuracy: {:.2f} ± {:.2f}".format(*(100 * x for x in mean_ci([correct(r) for r in rows]))))
if args.benchmark == "zeshel":
    print("normalized accuracy: {:.2f} ± {:.2f}".format(*(100 * x for x in mean_ci([correct(r) for r in rows if in_candidates(r)]))))
    for c in CATEGORIES:
        print(f"  {c}: " + "{:.2f} ± {:.2f}".format(*(100 * x for x in mean_ci([correct(r) for r in rows if r["category"] == c and in_candidates(r)]))))

if args.top_k:
    hits = [(r["ground_truth"] in ranked(r["metadata"]["mentions"][0]["scores"], args.top_k), in_candidates(r)) for r in rows if r["metadata"].get("mentions")]
    print(f"acc@{args.top_k}: {100 * np.mean([h for h, _ in hits]):.2f}")
    if args.benchmark == "zeshel":
        print(f"normalized acc@{args.top_k}: {100 * np.mean([h for h, ok in hits if ok]):.2f}")

if args.sc:
    examples = list(samples(rows))
    n = min(len(p) for _, p in examples)
    for k in range(1, n + 1):
        p = np.mean([pass_at_k(len(preds), sum(x == g for x in preds), k) for g, preds in examples])
        m = np.mean([maj_at_k(g, preds, k) for g, preds in examples])
        print(f"k={k} pass@k={100 * p:.2f} maj@k={100 * m:.2f}")
