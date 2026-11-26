import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ed import add_linker_args, build_linker, lookup, nlp, ranked, read_jsonl, restrict, resume

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("domain", choices=["general", "biomedical", "scientific"])
    parser.add_argument("output")
    add_linker_args(parser)
    args = parser.parse_args()

    kb = {k: [c[0] for c in v] for k, v in json.load(open(Path(args.data) / "acronym_kb.json")).items()}
    shortlists = [ranked(r["metadata"]["mentions"][0]["scores"], args.top_k) for r in read_jsonl(args.shortlist)] if args.shortlist else None
    linker = build_linker(args)
    lines = open(Path(args.data) / args.domain / "test.json", encoding="utf-8").readlines()
    end = len(lines) if args.n_docs is None else min(len(lines), args.n_docs)

    with open(args.output, "a", encoding="utf-8", buffering=1) as out:
        for i in tqdm(range(resume(args.output), end)):
            d = json.loads(lines[i])
            text = " ".join(d["tokens"])
            start = text.find(d["short_term"])
            if start == -1:
                continue
            doc = nlp(text)
            span = doc.char_span(start, start + len(d["short_term"]), alignment_mode="expand")
            candidates = lookup(kb, d["short_term"])
            cleaned = restrict(candidates, shortlists[i], args.shuffle) if shortlists else None
            try:
                predictions, meta = linker(doc, [(span, candidates if cleaned is None else cleaned)])
            except Exception:
                predictions, meta = [""], {}
            if cleaned is not None:
                meta["cleaned_candidates"] = cleaned
            record = {
                "prediction": predictions[0],
                "short_term": d["short_term"],
                "long_term": d["long_term"],
                "candidates": candidates,
                "metadata": meta,
            }
            out.write(json.dumps(record) + "\n")
