# **Fine-Tuning DeepSeek-R1 on Scientific Text Chunks (Computational complexity)**


This guide documents the full pipeline for preparing and fine-tuning  **DeepSeek-R1** on custom scientific texts (e.g., research papers from arXiv in the field of computation complexity), using a consumer GPU like RTX 3060.

## Overview of the Pipeline

1. **Extract text from scientific PDFs**
2. **Chunk and preprocess text**
3. **Generate structured JSON with metadata and keywords**
4. **Convert data to a format suitable for fine-tuning**
5. **Fine-tune a causal LLM using LoRA on a local GPU**

## **1.  Extract and Chunk Text from PDF**

Python script used to:
- Read scientific PDFs
- Clean and chunk them into ~500 character sections
- Extract keywords using YAKE
- Store structured JSON for each chunk

**Key script:** `PdfToJSON.py`

```python
# chunking and keyword extraction logic
chunk_text()
extract_keywords()
structure_data()
```

Each JSON looks like this:

```json
{
  "id": "uuid",
  "title": "Doc Title",
  "chunk_text": "Scientific content...",
  "metadata": {
    "author": "Author",
    "keywords": ["Subset", "Sum", "NP-complete"]
  }
}
```
## 2. Convert Chunks to Fine-tuning Format

Since the original data often lacks meaningful titles, we generate **task-specific prompts** based on each chunk’s keywords to make the dataset more specialized.

We use a set of **prompt templates** (shown below) and insert the top keywords extracted from every chunk to produce a prompt tailored for that specific paragraph.
```python
SPECIALIZED_PROMPTS = [
    "Explain the relationship between {keywords} in computational complexity and their real-world implications.",
    "This discussion covers {keywords} in theoretical computer science, focusing on their interdependencies.",
    ...
    "Compare {keywords} in the context of polynomial hierarchies and structural complexity theory."
]
```

**Output format example**

```json
  {
    "prompt": "This discussion covers Proof, Lemma, esk, ΛkIkΘk, logγ, Denoting, probability, law, total, prove in theoretical computer science, focusing on their interdependencies.",
    "response": "■ B.4 Proof of Lemma B.3 Proof. Denoting d = logγ(∆0/b∆) + 2, by the law of total probability, we have P(B) = P ( Tϵ > T −1, T−1 X k=0 (ϑα + ek + esk) ≤(2ϵf + ϑα + 2s)T, T−1 X k=0 ΛkIkΘk ...
  },
```

**Script used:** `FineTuningData.py`

**Output File :** `fine_tune_data.json`


## 3. Fine-Tuning the Model (with LoRA)

 - Using `DeepSeek-R1` (or any LoRA-compatible causal model):

- ### Step 1: Load fine-tuning data
```python
with open("JSONFIles/fine_tune_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
```
- ### Step 2: Tokenizer & Model

```python
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure proper batching

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
```
- ### Step 3: PEFT (LoRA)
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
```
- ### Step 4: Tokenize dataset 
```python 
def tokenize(example):
    return tokenizer(
        f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}",
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize)
```
- ### Step 5: Training & Saving the model
```python 
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained('DeepSeek-R1-Computational-Complexity')
tokenizer.save_pretrained("DeepSeek-R1-Computational-Complexity")

```

## 3. Testing the FIneTuned Model (with LoRA)
 
```python
Prompt = "Generate a very deep question about computational complexity"
```


## **Noraml pretrained models answere :** 

```python
answere = "The generated questions must not have any markdown formatting and should be in plain text. They can't include any links or references beyond standard sources like Wikipedia.
Okay, so I need to come up with some really good questions that delve into the depths of computational complexity, focusing on time and space for algorithm resources. Hmm, where do I start?
 First off, maybe think about well-known problems because they're usually studied extensively. The P vs NP problem comes to mind since it's one of those fundamental questions everyone is interested in. But wait...."
```
## Fine-Tuned models answere : 
```python
answere = """theory, which can be used to test AI systems.

The question should involve concepts like P vs NP problem and time-space trade-offs.
</think>

Certainly! Here's an advanced question that combines the core of your request with recent developments in theoretical computer science:


**Question:**  
"Is there a proof or argument showing whether any algorithm exists for solving SAT in less than exponential time? If such algorithms exist, what is their fundamental reason?"

This question delves into critical areas of computational complexity, specifically addressing the famous P vs NP problem. It challenges current understanding by questioning the feasibility of achieving "sub-exponential" runtime for problems traditionally considered "NP-hard."""
```
---