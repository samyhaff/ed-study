# When Does Reasoning Help Entity Disambiguation? Small LLMs In-Domain and Under Shift

Code for our *Findings of EACL 2027* paper.

[Paper](https://arxiv.org/abs/2611.30334) · [Models](https://hf.co/collections/samyhaff/ed-study)

## Setup

```bash
uv sync
```

Data goes in `data/zelda` ([ZELDA](https://github.com/flairNLP/zelda)), `data/zeshel` ([ZESHEL](https://github.com/lajanugen/zeshel)) and `data/gladis` ([GLADIS](https://github.com/LihuChen/GLADIS)).

The teacher model is served with:

```bash
uv run trl vllm-serve --model google/gemma-4-26B-A4B-it --port 8000 --max_model_len 8192 --tensor_parallel_size 2 --gpu_memory_utilization 0.65
```

## Inference

Listwise paradigms rerank the top-10 shortlist produced by the pointwise reranker (`--shortlist`).

```bash
# ZELDA
uv run annotate_zelda.py data/zelda test_aida-b out/aida_reranker.jsonl --model <reranker> --reranker
uv run annotate_zelda.py data/zelda test_aida-b out/aida_direct.jsonl --model <direct-selector> --direct --shortlist out/aida_reranker.jsonl --max_model_len 32768
uv run annotate_zelda.py data/zelda test_aida-b out/aida_tts.jsonl --model <reasoner> --n 10 --shortlist out/aida_reranker.jsonl

# ZESHEL, GLADIS
uv run annotate_zeshel.py data/zeshel test out/lego_tts.jsonl --domains lego --model <reasoner> --n 10 --shortlist out/lego_reranker.jsonl --context_window 3072
uv run annotate_gladis.py data/gladis biomedical out/gladis_tts.jsonl --model <reasoner> --n 10 --shortlist out/gladis_reranker.jsonl --context_window 3072
```

```bash
uv run evaluate.py zelda out/aida_tts.jsonl --sc
uv run evaluate.py zeshel out/lego_tts.jsonl
uv run evaluate.py zeshel out/lego_reranker.jsonl --top_k 10
uv run evaluate.py zeshel out/lego_fewshot.jsonl --exclude_train_data data/fewshot/lego_direct.jsonl
uv run evaluate.py gladis out/gladis_tts.jsonl
```

## Training: classical setting (ZELDA)

```bash
uv run prepare_zelda.py data/zelda
uv run annotate_zelda.py data/zelda train out/train_reranker.jsonl --model tomaarsen/Qwen3-Reranker-4B-seq-cls --reranker
```

```bash
# Pointwise reranker
uv run build_zelda_data.py reranker data/zelda data/reranker.jsonl --shortlist out/train_reranker.jsonl
uv run train_reranker.py data/reranker.jsonl models/reranker-4b --model tomaarsen/Qwen3-Reranker-4B-seq-cls --quantize

# Direct selection (lr 1e-4 for 0.6B, 1e-5 otherwise)
uv run build_zelda_data.py direct data/zelda data/direct.jsonl --shortlist out/train_reranker.jsonl
uv run train_sft.py data/direct.jsonl models/direct-4b --model Qwen/Qwen3-4B --quantize

# Off-policy distillation (λ=0, β=0)
uv run annotate_zelda.py data/zelda train out/train_teacher.jsonl --model google/gemma-4-26B-A4B-it --n 10 --shortlist out/train_reranker.jsonl --shuffle
uv run build_zelda_data.py distill data/zelda data/distill.jsonl --predictions out/train_teacher.jsonl
uv run train_gkd.py data/distill.jsonl models/e4b-off http://localhost:8000 --student google/gemma-4-E4B-it
uv run train_sft.py data/distill.jsonl models/qwen-0.6b-off --model Qwen/Qwen3-0.6B --learning_rate 1e-4

# On-policy distillation (λ=1, β=1)
uv run train_gkd.py data/distill.jsonl models/e4b-on http://localhost:8000 --student google/gemma-4-E4B-it --lmbda 1 --beta 1

# RLVR (GRPO)
uv run annotate_zelda.py data/zelda train out/train_student.jsonl --model Qwen/Qwen3-4B --n 10 --shortlist out/train_reranker.jsonl --shuffle
uv run build_zelda_data.py grpo data/zelda data/grpo.jsonl --shortlist out/train_reranker.jsonl --predictions out/train_student.jsonl
uv run accelerate launch train_grpo.py data/grpo.jsonl models/qwen-4b-grpo --model Qwen/Qwen3-4B
```

Merge LoRA adapters before inference:

```bash
uv run merge_lora.py models/e4b-off/checkpoint-2700 models/e4b-off/model
```

## Training: few-shot setting (ZESHEL)

Per test domain, 100 mentions are sampled for training (10 held out for validation); the rest are used for evaluation.

```bash
uv run annotate_zeshel.py data/zeshel test out/lego_base_reranker.jsonl --domains lego --model tomaarsen/Qwen3-Reranker-4B-seq-cls --reranker
uv run build_zeshel_data.py reranker data/zeshel lego data/fewshot/lego_reranker.jsonl --shortlist out/lego_base_reranker.jsonl
uv run build_zeshel_data.py direct data/zeshel lego data/fewshot/lego_direct.jsonl --shortlist out/lego_base_reranker.jsonl
uv run build_zeshel_data.py grpo data/zeshel lego data/fewshot/lego_grpo.jsonl --shortlist out/lego_base_reranker.jsonl
uv run annotate_zeshel.py data/zeshel test out/lego_teacher.jsonl --domains lego --model google/gemma-4-26B-A4B-it --n 10 --shortlist out/lego_base_reranker.jsonl --context_window 3072 --shuffle
uv run build_zeshel_data.py distill data/zeshel lego data/fewshot/lego_distill.jsonl --predictions out/lego_teacher.jsonl
```

```bash
uv run train_reranker.py data/fewshot/lego_reranker.jsonl models/reranker-4b-lego --model tomaarsen/Qwen3-Reranker-4B-seq-cls --base_adapter models/reranker-4b/checkpoint-9000 --epochs 15 --eval_split 0.1 --save_steps 50 --eval_steps 50 --quantize
uv run train_sft.py data/fewshot/lego_direct.jsonl models/direct-4b-lego --model Qwen/Qwen3-4B --base_adapter models/direct-4b/checkpoint-7500 --epochs 100 --eval_split 0.1 --quantize
uv run train_gkd.py data/fewshot/lego_distill.jsonl models/e4b-off-lego http://localhost:8000 --student models/e4b-off/model --epochs 200 --eval_split 0.1
uv run accelerate launch train_grpo.py data/fewshot/lego_grpo.jsonl models/e4b-grpo-lego --model models/e4b-off/model --epochs 15 --per_device_train_batch_size 2 --eval_split 0.1
```

## Citation

```bibtex
@inproceedings{haffoudhi2027reasoning,
  title     = {When Does Reasoning Help Entity Disambiguation? Small {LLM}s In-Domain and Under Shift},
  author    = {Haffoudhi, Samy and Suchanek, Fabian M. and Holzenberger, Nils},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027},
  eprint    = {2611.30334},
  archivePrefix = {arXiv}
}
```
