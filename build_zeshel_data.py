import argparse
import random

from ed import domain_split, load_zeshel, mark, messages, read_jsonl, reranker_document, reranker_query, teacher_answer, trim, write_jsonl, ws_nlp

parser = argparse.ArgumentParser()
parser.add_argument("target", choices=["reranker", "direct", "grpo", "distill"])
parser.add_argument("data")
parser.add_argument("domain")
parser.add_argument("output")
parser.add_argument("--shortlist")
parser.add_argument("--predictions")
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--max_doc_len", type=int, default=3072)
parser.add_argument("--num_negatives", type=int, default=9)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

rng = random.Random(args.seed)
docs, mentions = load_zeshel(args.data, domain_split(args.domain), [args.domain])
scores = {}
if args.shortlist:
    for r in read_jsonl(args.shortlist):
        if r.get("metadata", {}).get("mentions"):
            scores[r["mention_id"]] = [t for t, _ in sorted(r["metadata"]["mentions"][0]["scores"], key=lambda x: x[1], reverse=True)]
predictions = {r["mention_id"]: r for r in read_jsonl(args.predictions)} if args.predictions else {}


def truncate(text, start_index, end_index):
    spans = [(t.idx, t.idx + len(t)) for t in ws_nlp(text)]
    s, e = spans[start_index][0], spans[end_index][1]
    if len(spans) <= args.max_doc_len:
        return mark(text, s, e)
    width = end_index - start_index + 1
    if width >= args.max_doc_len:
        left, right = start_index, end_index + 1
    else:
        budget = args.max_doc_len - width
        left, right = start_index - budget // 2, end_index + 1 + budget - budget // 2
        if left < 0:
            right, left = right - left, 0
        if right > len(spans):
            left, right = max(0, left - (right - len(spans))), len(spans)
    cut = spans[left][0]
    return mark(text[cut : spans[right - 1][1]], s - cut, e - cut)


def shortlist(m, gold):
    title_to_id = {}
    for c in m.tfidf_candidates:
        title_to_id.setdefault(docs.loc[c, "title"], c)
    ranking = scores.get(m.Index, [])
    titles = ([t for t in ranking if t in title_to_id] if ranking else list(title_to_id))[: args.top_k]
    if gold not in titles:
        if gold not in title_to_id:
            match = docs[docs["title"] == gold]
            if len(match):
                title_to_id[gold] = match.index[0]
        if gold in title_to_id:
            if titles:
                titles[rng.randrange(len(titles))] = gold
            else:
                titles.append(gold)
    return [(t, trim(docs.loc[title_to_id[t], "text"])) for t in titles]


rows = []
ids = list(mentions.index)
rng.shuffle(ids)
for mention_id in ids[: args.n_samples]:
    if args.target == "distill":
        r = predictions.get(mention_id)
        if r is None or not r.get("metadata", {}).get("mentions") or not r["metadata"].get("cleaned_candidates"):
            continue
        candidates, m = r["metadata"]["cleaned_candidates"], r["metadata"]["mentions"][0]
        g = next((i + 1 for i, c in enumerate(candidates) if c[0] == r["ground_truth"]), None)
        correct = [o for o in m["model_outputs"] if g is not None and teacher_answer(o) == g]
        if correct:
            text = mark(r["metadata"]["context"], m["mention_start_char"], m["mention_end_char"])
            rows.append({"id": mention_id, "messages": messages(text, candidates) + [{"role": "assistant", "content": rng.choice(correct)}]})
        continue
    m = next(mentions.loc[[mention_id]].itertuples())
    text = truncate(docs.loc[m.context_document_id, "text"], int(m.start_index), int(m.end_index))
    gold = docs.loc[m.label_document_id, "title"]
    candidates = shortlist(m, gold)
    titles = [c[0] for c in candidates]
    if args.target == "reranker":
        description = dict(candidates).get(gold)
        if description is None:
            continue
        query = reranker_query(text)
        rows.append({"text": query + reranker_document(gold, description), "label": 1.0, "mention_id": mention_id})
        pool = [t for t in scores.get(mention_id, []) or titles if t != gold]
        negatives = [t for t in dict.fromkeys(pool) if t in dict(candidates)][: args.num_negatives]
        rows += [{"text": query + reranker_document(t, dict(candidates)[t]), "label": 0.0, "mention_id": mention_id} for t in negatives]
    elif gold in titles:
        rng.shuffle(candidates)
        titles = [c[0] for c in candidates]
        rows.append({"mention_id": mention_id, "corpus": m.corpus, "prompt": messages(text, candidates), "answer": titles.index(gold) + 1})

rng.shuffle(rows)
write_jsonl(args.output, rows)
