import json
import pickle
import random
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer

PROMPT = """You are an expert designed to disambiguate entities in text, taking into account the overall context and a list of entity candidates.
You are provided with an input text that includes a full contextual narrative, the mention enclosed between the '[' and ']' characters, and a list of candidates, each preceded by an index number.
Your task is to determine the most appropriate entity from the candidates based on the context and candidate entity descriptions.
Please show your choice in the answer field with only the choice index number, e.g., "answer": 3."""

RERANKER_TASK = "Given a text with a mention enclosed between the '[' and ']' characters, retrieve relevant entities that the mention refers to."
RERANKER_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

TRAIN_DOMAINS = ["american_football", "doctor_who", "fallout", "final_fantasy", "military", "pro_wrestling", "starwars", "world_of_warcraft"]
EVAL_DOMAINS = ["coronation_street", "muppets", "elder_scrolls", "ice_hockey"]
TEST_DOMAINS = ["yugioh", "star_trek", "lego", "forgotten_realms"]

PUNC = re.compile(r"[\W]+")

ws_nlp = spacy.blank("en")
ws_nlp.tokenizer = Tokenizer(ws_nlp.vocab, token_match=re.compile(r"\S+").match)
ws_nlp.add_pipe("sentencizer")
nlp = spacy.blank("en")


def mark(text, start, end):
    return f"{text[:start]}[{text[start:end]}]{text[end:]}"


def reranker_query(text):
    return f"{RERANKER_PREFIX}<Instruct>: {RERANKER_TASK}\n<Query>: {text}\n"


def reranker_document(title, description):
    return f"<Document>: {f'{title} ({description})'.strip()}{RERANKER_SUFFIX}"


def messages(text, candidates):
    user = f"Input text: {text}"
    if candidates:
        user += "\nList of candidate entities:\n" + "\n".join(f"{i + 1}. {t} - {d}" for i, (t, d) in enumerate(candidates))
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": user}]


def parse_answer(output):
    matches = re.findall(r"answer\D*:\s*(\d+)", output, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
    if lines:
        numbers = re.findall(r"\d+", lines[-1])
        if numbers:
            return int(numbers[-1])
    return 0


def teacher_answer(output, fallback=True):
    for pattern in (r'thought\s*(.*?)(?:```json\s*)?\{\s*"answer"\s*:\s*(\d+)\s*\}', r"^(.*?)</think>.*?(\d+)\s*$"):
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            return int(match.group(2))
    return parse_answer(output) if fallback else None


def vote(outputs, n_candidates):
    answers = [a for a in map(parse_answer, outputs) if 0 < a <= n_candidates]
    return Counter(answers).most_common(1)[0][0] if answers else 0


def ranked(scores, k):
    return [title for title, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]


def restrict(candidates, shortlist, shuffle=False):
    rank = {title: i for i, title in enumerate(shortlist)}
    out = sorted([c for c in candidates if c[0] in rank], key=lambda c: rank[c[0]])
    if shuffle:
        random.shuffle(out)
    return out


def alias_table(counter):
    table = {}
    for mention, entities in counter.items():
        table.setdefault(mention.replace(" ", "").lower(), set()).update(entities.keys())
        table.setdefault(PUNC.sub("", mention.lower()), set()).update(entities.keys())
    return {k: list(v) for k, v in table.items()}


def lookup(table, mention, descriptions=None):
    found = set()
    for key in (mention, mention.replace(" ", "").lower(), PUNC.sub("", mention.lower())):
        found.update(table.get(key, []))
    if not descriptions:
        return [(c, "") for c in found]
    out = [(c, descriptions.get(c) or "") for c in found]
    random.shuffle(out)
    return out


def load_zelda_kb(data_dir):
    with open(Path(data_dir) / "other/zelda_mention_entities_counter.pickle", "rb") as f:
        table = alias_table(pickle.load(f))
    descriptions = {}
    with open(Path(data_dir) / "other/entity_descriptions.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            descriptions[d["wikipedia_title"]] = d["description"]
    return table, descriptions


def zelda_path(data_dir, split):
    if split in ("train", "val"):
        return Path(data_dir) / f"{split}_sample.jsonl"
    return Path(data_dir) / f"test_data/jsonl/{split}.jsonl"


def trim(description, n=3):
    return " ".join(s.text for s in list(ws_nlp(description).sents)[:n])


def load_zeshel(data_dir, split, domains):
    docs = pd.concat([pd.read_json(Path(data_dir) / f"documents/{d}.json", lines=True) for d in domains], ignore_index=True).set_index("document_id")
    mentions = pd.read_json(Path(data_dir) / f"mentions/{split}.json", lines=True).set_index("mention_id")
    mentions = mentions[mentions["corpus"].isin(domains)]
    candidates = pd.read_json(Path(data_dir) / f"tfidf_candidates/{split}.json", lines=True).set_index("mention_id")
    return docs, mentions.join(candidates, how="inner")


def zeshel_span(text, start_index, end_index):
    spans = [(t.idx, t.idx + len(t)) for t in ws_nlp(text)]
    return spans[start_index][0], spans[end_index][1]


def domain_split(domain):
    return "train" if domain in TRAIN_DOMAINS else "val" if domain in EVAL_DOMAINS else "test"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Reranker:
    def __init__(self, model):
        from vllm import LLM

        self.llm = LLM(model=model, runner="pooling", trust_remote_code=True)

    def __call__(self, doc, items):
        start = time.perf_counter()
        predictions, meta = [], []
        for span, candidates in items:
            scores, best = [], ""
            if candidates:
                query = reranker_query(mark(doc.text, span.start_char, span.end_char))
                outputs = self.llm.score(query, [reranker_document(t, d) for t, d in candidates])
                scores = [o.outputs.score for o in outputs]
                best = candidates[scores.index(max(scores))][0]
            predictions.append(best)
            meta.append(
                {
                    "mention_text": span.text,
                    "mention_start_char": span.start_char,
                    "mention_end_char": span.end_char,
                    "prediction": best,
                    "scores": list(zip([c[0] for c in candidates], scores)),
                }
            )
        return predictions, {"mentions": meta, "context": doc.text, "disambiguation_time": time.perf_counter() - start}


class Selector:
    def __init__(self, model, direct=False, n=10, max_tokens=2048, max_model_len=None, context_window=0, tensor_parallel_size=1):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
        defaults = self.llm.model_config.get_diff_sampling_param()
        self.params = SamplingParams(**(defaults | {"max_tokens": max_tokens, "n": 1 if direct else n}))
        self.thinking = not direct
        self.context_window = context_window

    def text(self, doc, span):
        if self.context_window <= 0:
            return mark(doc.text, span.start_char, span.end_char)
        left = max(0, span.start - self.context_window // 2)
        right = min(len(doc), span.end + self.context_window // 2)
        return f"{doc[left:span.start].text} [{span.text}] {doc[span.end:right].text}".strip()

    def __call__(self, doc, items):
        start = time.perf_counter()
        queued = [i for i, (_, c) in enumerate(items) if len(c) > 1]
        try:
            outputs = self.llm.chat(
                [messages(self.text(doc, items[i][0]), items[i][1]) for i in queued],
                sampling_params=self.params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": self.thinking},
            )
        except Exception:
            outputs = [None] * len(queued)
        responses = dict(zip(queued, outputs))
        predictions, meta = [], []
        for i, (span, candidates) in enumerate(items):
            m = {"model_outputs": [], "tokens_in": 0, "tokens_out": []}
            if i not in responses:
                label = candidates[0][0] if candidates else ""
            elif responses[i] is None:
                label = ""
            else:
                r = responses[i]
                answer = vote([o.text for o in r.outputs], len(candidates))
                label = candidates[answer - 1][0] if answer else ""
                m = {"model_outputs": [o.text for o in r.outputs], "tokens_in": len(r.prompt_token_ids), "tokens_out": [len(o.token_ids) for o in r.outputs]}
            label = label.strip()
            predictions.append(label)
            meta.append(m | {"mention_text": span.text, "mention_start_char": span.start_char, "mention_end_char": span.end_char, "prediction": label})
        return predictions, {"mentions": meta, "context": doc.text, "disambiguation_time": time.perf_counter() - start}


def add_linker_args(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--shortlist")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--context_window", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--n_docs", type=int)


def build_linker(args):
    random.seed(42)
    if args.reranker:
        return Reranker(args.model)
    return Selector(args.model, args.direct, args.n, args.max_tokens, args.max_model_len, args.context_window, args.tensor_parallel_size)


def resume(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return sum(1 for _ in open(p)) if p.exists() else 0
