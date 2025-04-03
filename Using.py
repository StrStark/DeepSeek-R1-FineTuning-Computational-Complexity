from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = "fine_tuned_model"
torch.backends.cuda.enable_flash_sdp(True)  # Enable FlashAttention
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Alternative optimization
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
model.config.use_sliding_window_attention = False

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure proper batching

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id

input_text = "Explain the difference between P, NP, and PSPACE. Then, given a decision problem X, describe the steps to determine whether X is NP-complete. Finally, analyze whether the following problem belongs to P, NP, or PSPACE: Given a Boolean formula in Conjunctive Normal Form (CNF) with n variables and m clauses, determine whether there exists an assignment that satisfies at least (m/2) clauses."
encoded_input = tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True).to(model.device)
output = model.generate(encoded_input, max_length=500, num_return_sequences=1)

print(tokenizer.decode(output[0], skip_special_tokens=True))
