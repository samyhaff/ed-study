import argparse

import numpy as np
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl.experimental.distillation import DistillationConfig, DistillationTrainer

_prepare_inputs = DistillationTrainer._prepare_inputs
DistillationTrainer._prepare_inputs = lambda self, batch: self._prepare_input(_prepare_inputs(self, batch))

parser = argparse.ArgumentParser()
parser.add_argument("train_data")
parser.add_argument("output_dir")
parser.add_argument("teacher_url")
parser.add_argument("--student", default="google/gemma-4-E4B-it")
parser.add_argument("--teacher", default="google/gemma-4-26B-A4B-it")
parser.add_argument("--lmbda", type=float, default=0.0)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
parser.add_argument("--length_quantile", type=float, default=0.9)
parser.add_argument("--max_length", type=int, default=8192)
parser.add_argument("--max_completion_length", type=int, default=2048)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.3)
parser.add_argument("--eval_split", type=float, default=0.01)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

dataset = load_dataset("json", data_files=args.train_data, split="train").select_columns(["messages"])
if "gemma" in args.student.lower():
    dataset = dataset.map(lambda x: {"messages": [x["messages"][0] | {"content": "<|think|>\n\n" + x["messages"][0]["content"]}] + x["messages"][1:]})

tokenizer = AutoTokenizer.from_pretrained(args.student)
dataset = dataset.filter(
    lambda x: len(tokenizer.apply_chat_template(x["messages"][:-1], add_generation_prompt=True, tokenize=True, return_dict=False)) <= args.max_length - args.max_completion_length,
    num_proc=8,
)
if args.length_quantile < 1.0:
    lengths = np.array(dataset.map(lambda x: {"n": len(tokenizer.apply_chat_template(x["messages"], tokenize=True, return_dict=False))}, num_proc=8)["n"])
    dataset = dataset.select(np.where(lengths <= int(np.quantile(lengths, args.length_quantile)))[0])
train = dataset.train_test_split(test_size=args.eval_split, seed=42)["train"]

config = DistillationConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    lr_scheduler_type="linear",
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    lmbda=args.lmbda,
    beta=args.beta,
    loss_top_k=1,
    loss_add_tail=True,
    temperature=args.temperature,
    use_teacher_server=True,
    teacher_model_server_url=args.teacher_url,
    teacher_model_init_kwargs={"dtype": "bfloat16"},
    model_init_kwargs={"dtype": "bfloat16"},
    max_length=args.max_length,
    max_completion_length=args.max_completion_length,
    use_vllm=args.lmbda > 0,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=args.gpu_memory_utilization,
    vllm_max_model_length=args.max_length,
    vllm_enable_sleep_mode=True,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1,
    save_steps=args.save_steps,
    eval_strategy="no",
)
peft = LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=256, lora_dropout=0.0, target_modules="all-linear", bias="none")
trainer = DistillationTrainer(model=args.student, teacher_model=args.teacher, args=config, train_dataset=train, peft_config=peft)
trainer.train(resume_from_checkpoint=args.resume)
trainer.save_model()
