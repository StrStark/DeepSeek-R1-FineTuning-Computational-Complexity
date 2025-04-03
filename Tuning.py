from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import torch
import json

# ---------- Step 1: Load fine-tuning data ----------
with open("JSONFIles/fine_tune_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# ---------- Step 2: Tokenizer & Model ----------s
# Change model identifier to DeepSeek-R1-Distill-Qwen-1.5B
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure proper batching

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
# ---------- Step 3: PEFT (LoRA) ----------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# ---------- Step 4: Tokenize dataset ----------
def tokenize(example):
    return tokenizer(
        f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}",
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize)

# ---------- Step 5: Training Arguments ----------
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    logging_dir="./logs",
    save_total_limit=2,
    report_to="none"
)

# ---------- Step 6: Trainer ----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained("fine_tuned_model")
