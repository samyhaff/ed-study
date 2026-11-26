import argparse
import json

from tqdm import tqdm

from ed import EVAL_DOMAINS, TEST_DOMAINS, TRAIN_DOMAINS, add_linker_args, build_linker, load_zeshel, nlp, ranked, read_jsonl, restrict, resume, trim, zeshel_span

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("split", choices=["train", "val", "test"])
    parser.add_argument("output")
    parser.add_argument("--domains", nargs="*", choices=TRAIN_DOMAINS + EVAL_DOMAINS + TEST_DOMAINS)
    parser.add_argument("--exclude_train_data")
    add_linker_args(parser)
    args = parser.parse_args()

    domains = args.domains or {"train": TRAIN_DOMAINS, "val": EVAL_DOMAINS, "test": TEST_DOMAINS}[args.split]
    docs, mentions = load_zeshel(args.data, args.split, domains)
    excluded = {r["mention_id"] for r in read_jsonl(args.exclude_train_data) if "mention_id" in r} if args.exclude_train_data else set()
    shortlists = {r["mention_id"]: ranked(r["metadata"]["mentions"][0]["scores"], args.top_k) for r in read_jsonl(args.shortlist) if r.get("metadata", {}).get("mentions")} if args.shortlist else {}
    linker = build_linker(args)
    done, processed = resume(args.output), 0

    with open(args.output, "a", encoding="utf-8", buffering=1) as out:
        for m in tqdm(mentions.itertuples(), total=len(mentions)):
            if m.Index in excluded:
                continue
            if processed < done:
                processed += 1
                continue
            if args.n_docs is not None and processed >= args.n_docs:
                break
            text = docs.loc[m.context_document_id, "text"]
            doc = nlp(text)
            span = doc.char_span(*zeshel_span(text, m.start_index, m.end_index), alignment_mode="expand")
            if span is None:
                continue
            candidates = [(docs.loc[c, "title"], trim(docs.loc[c, "text"])) for c in m.tfidf_candidates]
            cleaned = restrict(candidates, shortlists[m.Index], args.shuffle) if m.Index in shortlists else None
            try:
                predictions, meta = linker(doc, [(span, candidates if cleaned is None else cleaned)])
            except Exception:
                predictions, meta = [""], {}
            if cleaned is not None:
                meta["cleaned_candidates"] = cleaned
            record = {
                "prediction": predictions[0],
                "ground_truth": docs.loc[m.label_document_id, "title"],
                "category": m.category,
                "corpus": m.corpus,
                "candidates": candidates,
                "mention_id": m.Index,
                "metadata": meta,
            }
            out.write(json.dumps(record) + "\n")
            processed += 1
