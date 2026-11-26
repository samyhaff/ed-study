import argparse
import json
import pickle
import random
import re
from pathlib import Path

from ed import alias_table, lookup, write_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("--max_doc_len", type=int, default=3072)
parser.add_argument("--min_candidates", type=int, default=10)
parser.add_argument("--n_mentions", type=int, default=55000)
parser.add_argument("--n_val", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

data = Path(args.data)
with open(data / "other/zelda_mention_entities_counter.pickle", "rb") as f:
    table = alias_table(pickle.load(f))

keep_fields = ("index", "wikipedia_titles", "wikipedia_ids")
docs = []
for line in open(data / "train_data/zelda_train.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if len(re.findall(r"\S+", d["text"])) >= args.max_doc_len:
        continue
    keep = []
    for i, ((s, e), gold) in enumerate(zip(d["index"], d["wikipedia_titles"])):
        candidates = [c for c, _ in lookup(table, d["text"][s:e].lower())]
        if len(candidates) >= args.min_candidates and gold in candidates:
            keep.append(i)
    if keep:
        docs.append(d | {k: [d[k][i] for i in keep] for k in keep_fields if k in d})

rng = random.Random(args.seed)
order = list(range(len(docs)))
rng.shuffle(order)
chosen = {i: rng.randrange(len(docs[i]["index"])) for i in order[: args.n_mentions]}
sample = [docs[i] | {k: [docs[i][k][j]] for k in keep_fields if k in docs[i]} for i, j in sorted(chosen.items())]
rng.shuffle(sample)
write_jsonl(data / "val_sample.jsonl", sample[: args.n_val])
write_jsonl(data / "train_sample.jsonl", sample[args.n_val :])
