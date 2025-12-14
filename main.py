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
    dtype=torch.bfloat16,
    device_map="auto"
)

# reference model（冻结）
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": "Give me a short introduction to large language model."
    },
    {
        "role": "user",
        "content": "hello world"
    }
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
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

dpo_config = DPOConfig(
    output_dir="./qwen3-dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    beta=0.1,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    max_length=2048,
    max_prompt_length=512,
    report_to="none"
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    #tokenizer=tokenizer,
)

trainer.train()
trainer.save_model()





