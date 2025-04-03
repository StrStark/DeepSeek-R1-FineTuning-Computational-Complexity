import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Use 4-bit quantization for RTX 3060 6GB
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for efficiency
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

input_text = "Generate a very deep question about computational complexity"

encoded_input = tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True).to(model.device)

output = model.generate(
    encoded_input,
    max_length=800,
    min_length=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    eos_token_id=model.config.eos_token_id,
    length_penalty=1.0,
    early_stopping=True
)

os.system("clear")

with open('Answere.md', 'w') as file:
    file.write(f"{input_text} \n --- \n"+tokenizer.decode(output[0], skip_special_tokens=True).replace(input_text , "").strip())
