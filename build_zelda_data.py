import argparse
import json
import random

from ed import load_zelda_kb, lookup, mark, messages, parse_answer, ranked, read_jsonl, reranker_document, reranker_query, restrict, teacher_answer, write_jsonl, zelda_path

parser = argparse.ArgumentParser()
parser.add_argument("target", choices=["reranker", "direct", "grpo", "distill"])
parser.add_argument("data")
parser.add_argument("output")
parser.add_argument("--shortlist")
parser.add_argument("--predictions")
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--num_negatives", type=int, default=9)
parser.add_argument("--max_pass", type=float, default=0.7)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
rng = random.Random(args.seed)
rows = []


def gold_index(candidates, gold):
    return next((i for i, c in enumerate(candidates) if c[0] == gold), None)


if args.target == "reranker":
    for r in read_jsonl(args.shortlist):
        gold, meta = r["ground_truths"][0], r["metadata"]
        m = meta["mentions"][0]
        descriptions = {c[0]: c[1] for c in r["candidates"][0] if len(c) == 2}
        if not descriptions.get(gold):
            continue
        query = reranker_query(mark(meta["context"], m["mention_start_char"], m["mention_end_char"]))
        rows.append({"text": query + reranker_document(gold, descriptions[gold]), "label": 1.0})
        negatives = [t for t in ranked(m["scores"], len(m["scores"])) if t != gold and descriptions.get(t)]
        rows += [{"text": query + reranker_document(t, descriptions[t]), "label": 0.0} for t in negatives[: args.num_negatives]]
    rng.shuffle(rows)

elif args.target == "distill":
    for r in read_jsonl(args.predictions):
        context = r["metadata"]["context"]
        for m, gold, candidates in zip(r["metadata"]["mentions"], r["ground_truths"], r["metadata"]["cleaned_candidates"]):
            g = gold_index(candidates, gold)
            correct = [o for o in m["model_outputs"] if g is not None and teacher_answer(o, False) == g + 1]
            if correct:
                text = mark(context, m["mention_start_char"], m["mention_end_char"])
                rows.append({"id": r["id"], "messages": messages(text, candidates) + [{"role": "assistant", "content": rng.choice(correct)}]})

else:
    table, descriptions = load_zelda_kb(args.data)
    shortlists = [{m["mention_start_char"]: ranked(m["scores"], args.top_k) for m in r["metadata"]["mentions"]} for r in read_jsonl(args.shortlist)]
    band = None
    if args.predictions:
        band = set()
        for r in read_jsonl(args.predictions):
            for i, m in enumerate(r["metadata"]["mentions"]):
                candidates = r["metadata"]["cleaned_candidates"][i]
                g = gold_index(candidates, r["ground_truths"][i].replace("_", " "))
                n = sum(parse_answer(o) == g + 1 for o in m["model_outputs"]) if g is not None else 0
                if 0 < n <= len(m["model_outputs"]) * args.max_pass:
                    band.add((r["id"], m["mention_start_char"], m["mention_end_char"]))
    for i, line in enumerate(open(zelda_path(args.data, "train"), encoding="utf-8")):
        d = json.loads(line)
        for gold, (s, e) in zip(d["wikipedia_titles"], d["index"]):
            if band is not None and (d["page_id"], s, e) not in band:
                continue
            candidates = restrict(lookup(table, d["text"][s:e], descriptions), shortlists[i].get(s, []), shuffle=True)
            if gold_index(candidates, gold) is None:
                continue
            text = mark(d["text"], s, e)
            g = gold_index(candidates, gold)
            rows.append({"prompt": messages(text, candidates), "answer": g + 1, "id": d["page_id"]})

write_jsonl(args.output, rows)
