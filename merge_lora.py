import argparse
import glob
import json
import os
import re
import shutil

import torch
from huggingface_hub import snapshot_download
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer

AUX_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "chat_template.jinja", "processor_config.json", "preprocessor_config.json")


def tensors(path):
    out = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out


def surgical_merge(adapter, output, base, config):
    scaling = config.lora_alpha / config.r
    weights = {}
    for shard in sorted(glob.glob(os.path.join(base, "*.safetensors"))):
        weights |= tensors(shard)
    pairs, overwrites = {}, {}
    for key, tensor in tensors(os.path.join(adapter, "adapter_model.safetensors")).items():
        if m := re.match(r"^base_model\.model\.(.+)\.(lora_A|lora_B)\.weight$", key):
            pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = tensor
        elif m := re.match(r"^base_model\.model\.(.+)\.modules_to_save\.default\.(.+)$", key):
            overwrites[m.group(1) + "." + m.group(2)] = tensor
        elif ".original_module." not in key:
            raise ValueError(key)
    for key, ab in pairs.items():
        w = weights[key]
        weights[key] = (w.float() + scaling * (ab["lora_B"].float() @ ab["lora_A"].float())).to(w.dtype)
    for key, tensor in overwrites.items():
        weights[key] = tensor.to(weights[key].dtype) if key in weights else tensor
    os.makedirs(output, exist_ok=True)
    save_file(weights, os.path.join(output, "model.safetensors"), metadata={"format": "pt"})
    for name in AUX_FILES:
        if os.path.exists(os.path.join(base, name)):
            shutil.copy(os.path.join(base, name), os.path.join(output, name))


def peft_merge(adapter, output, base, config):
    if str(config.task_type).endswith("SEQ_CLS"):
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    PeftModel.from_pretrained(model, adapter).merge_and_unload().save_pretrained(output)
    AutoTokenizer.from_pretrained(base).save_pretrained(output)
    try:
        AutoProcessor.from_pretrained(base).save_pretrained(output)
    except Exception:
        pass


parser = argparse.ArgumentParser()
parser.add_argument("adapter")
parser.add_argument("output")
parser.add_argument("--base_model")
args = parser.parse_args()

config = PeftConfig.from_pretrained(args.adapter)
base = args.base_model or config.base_model_name_or_path
base = base if os.path.isdir(base) else snapshot_download(base)
with open(os.path.join(base, "config.json")) as f:
    model_type = json.load(f).get("model_type", "")
(surgical_merge if "gemma4" in model_type else peft_merge)(args.adapter, args.output, base, config)
