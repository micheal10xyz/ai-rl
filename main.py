from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datasets
from trl import DPOTrainer, DPOConfig

model_name = "/Users/shiqining/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# policy model（可训练）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto"
)

# reference model（冻结）
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto"
)

rl_dataset = datasets.load_dataset("json", data_files="helpful_base_cn_train.jsonl")


def process_message(message):
    role_name = message["role"]
    text = message["text"]
    role_name = "user" if role_name == "human" else role_name
    return {
        "role": role_name,
        "content": text
    }


def pre_process(example):
    context = example["context"]
    messages = [process_message(message) for message in context]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {
        "prompt": prompt,
        "chosen": example["chosen"]["text"],
        "rejected": example["rejected"]["text"]
    }

train_dataset = rl_dataset["train"].map(pre_process)
train_dataset = train_dataset.select(range(100))
training_args = DPOConfig(output_dir="qwen3-dpo")
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)


trainer.train()
trainer.save_model()