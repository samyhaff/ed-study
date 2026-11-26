import argparse
import json

from tqdm import tqdm

from ed import add_linker_args, build_linker, load_zelda_kb, lookup, nlp, ranked, read_jsonl, restrict, resume, zelda_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("split")
    parser.add_argument("output")
    add_linker_args(parser)
    args = parser.parse_args()

    table, descriptions = load_zelda_kb(args.data)
    shortlists = [{m["mention_start_char"]: ranked(m["scores"], args.top_k) for m in r["metadata"]["mentions"]} for r in read_jsonl(args.shortlist)] if args.shortlist else None
    linker = build_linker(args)
    lines = open(zelda_path(args.data, args.split), encoding="utf-8").readlines()
    end = len(lines) if args.n_docs is None else min(len(lines), args.n_docs)

    with open(args.output, "a", encoding="utf-8", buffering=1) as out:
        for i in tqdm(range(resume(args.output), end)):
            d = json.loads(lines[i])
            pairs = sorted(zip(d["index"], d["wikipedia_titles"]), key=lambda x: x[0][0])
            doc = nlp(d["text"])
            items, full = [], []
            for (s, e), _ in pairs:
                span = doc.char_span(s, e, alignment_mode="expand")
                candidates = lookup(table, span.text, descriptions)
                full.append(candidates)
                items.append((span, restrict(candidates, shortlists[i][span.start_char], args.shuffle) if shortlists else candidates))
            try:
                predictions, meta = linker(doc, items)
            except Exception:
                predictions, meta = [""] * len(items), {}
            if shortlists:
                meta["cleaned_candidates"] = [c for _, c in items]
            record = {
                "id": d["page_id"] if args.split in ("train", "val") else d["id"],
                "ground_truths": [g for _, g in pairs],
                "predictions": predictions,
                "candidates": full,
                "metadata": meta,
            }
            out.write(json.dumps(record) + "\n")
