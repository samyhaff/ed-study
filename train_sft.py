import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("train_data")
parser.add_argument("output_dir")
parser.add_argument("--model", default="Qwen/Qwen3-4B")
parser.add_argument("--base_adapter")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--max_length", type=int)
parser.add_argument("--eval_split", type=float, default=0.01)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
dataset = load_dataset("json", data_files=args.train_data, split="train")
traces = "messages" in dataset.column_names
if traces:
    dataset = dataset.select_columns(["messages"])
else:
    dataset = dataset.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False),
            "completion": f'"answer": {x["answer"]}',
        },
        remove_columns=dataset.column_names,
    )
train, test = (dataset, None) if args.eval_split <= 0 else dataset.train_test_split(test_size=args.eval_split, seed=42).values()

quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16) if args.quantize else None
model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto", quantization_config=quantization)
if args.quantize:
    model = prepare_model_for_kbit_training(model)
if args.base_adapter:
    model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=256, lora_dropout=0.0, target_modules="all-linear", bias="none"))

config = SFTConfig(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    lr_scheduler_type="linear",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size if traces else 1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_length=args.max_length or (8192 if traces else 5120),
    assistant_only_loss=traces,
    logging_steps=1,
    save_steps=args.save_steps,
    eval_strategy="no" if test is None else "steps",
    eval_steps=args.eval_steps,
    eval_on_start=test is not None,
)
trainer = SFTTrainer(model=model, args=config, train_dataset=train, eval_dataset=test, processing_class=tokenizer)
trainer.train(resume_from_checkpoint=args.resume)
trainer.save_model(f"{args.output_dir}/final_model")
