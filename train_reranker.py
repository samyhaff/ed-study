import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding, Trainer, TrainingArguments


class RerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits.view(-1), labels.float().view(-1))
        return (loss, outputs) if return_outputs else loss


parser = argparse.ArgumentParser()
parser.add_argument("train_data")
parser.add_argument("output_dir")
parser.add_argument("--model", default="tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
parser.add_argument("--base_adapter")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
parser.add_argument("--max_length", type=int, default=4096)
parser.add_argument("--eval_split", type=float, default=0.01)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16) if args.quantize else None
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, dtype=torch.bfloat16, quantization_config=quantization)
model.config.pad_token_id = tokenizer.pad_token_id
if args.quantize:
    model = prepare_model_for_kbit_training(model)
if args.base_adapter:
    model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
else:
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=128,
        lora_alpha=256,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)

dataset = load_dataset("json", data_files=args.train_data, split="train")
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=args.max_length) | {"labels": x["label"]}, batched=True, remove_columns=dataset.column_names)
train, test = (dataset, None) if args.eval_split <= 0 else dataset.train_test_split(test_size=args.eval_split, seed=42).values()

config = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    lr_scheduler_type="linear",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.epochs,
    logging_steps=1,
    save_steps=args.save_steps,
    eval_strategy="no" if test is None else "steps",
    eval_steps=args.eval_steps,
    eval_on_start=test is not None,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
)
trainer = RerankerTrainer(model=model, args=config, train_dataset=train, eval_dataset=test, data_collator=DataCollatorWithPadding(tokenizer))
trainer.train(resume_from_checkpoint=args.resume)
trainer.save_model(f"{args.output_dir}/final_model")
