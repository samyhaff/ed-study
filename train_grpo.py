import argparse
import re

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def reward(prompts, completions, answer, **kwargs):
    found = [re.findall(r"answer\D*:\s*(\d+)", c[0]["content"], re.IGNORECASE) for c in completions]
    return [float(bool(m) and int(m[-1]) == a) for m, a in zip(found, answer)]


parser = argparse.ArgumentParser()
parser.add_argument("train_data")
parser.add_argument("output_dir")
parser.add_argument("--model", default="Qwen/Qwen3-4B")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_generations", type=int, default=16)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int)
parser.add_argument("--max_seq_length", type=int, default=8192)
parser.add_argument("--use_vllm", action="store_true")
parser.add_argument("--eval_split", type=float, default=0.0)
parser.add_argument("--save_steps", type=int, default=50)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
for name in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
    if hasattr(model.model, name):
        delattr(model.model, name)
model = get_peft_model(model, LoraConfig(r=128, lora_alpha=256, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
if "<turn|>" in tokenizer.get_vocab():
    tokenizer.eos_token = "<turn|>"

dataset = load_dataset("json", data_files=args.train_data, split="train")
lengths = np.array([len(ids) for ids in tokenizer.apply_chat_template(list(dataset["prompt"]), add_generation_prompt=True, tokenize=True, return_dict=False)])
max_prompt = int(np.quantile(lengths, 0.9))
dataset = dataset.select(np.where(lengths <= max_prompt)[0])
if args.eval_split > 0:
    dataset = dataset.train_test_split(test_size=args.eval_split, seed=42)["train"]

config = GRPOConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps or args.num_generations // args.per_device_train_batch_size,
    num_generations=args.num_generations,
    temperature=args.temperature,
    max_completion_length=args.max_seq_length - max_prompt - 1,
    loss_type="grpo",
    beta=0.0,
    epsilon=0.2,
    chat_template_kwargs={"enable_thinking": True},
    use_vllm=args.use_vllm,
    vllm_gpu_memory_utilization=0.5,
    vllm_max_model_length=args.max_seq_length,
    vllm_enable_sleep_mode=True,
    bf16=True,
    logging_steps=1,
    save_steps=args.save_steps,
)
trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[reward], args=config, train_dataset=dataset)
trainer.train(resume_from_checkpoint=args.resume)
