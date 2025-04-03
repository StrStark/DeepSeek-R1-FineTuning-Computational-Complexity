import os
import json
import random

DATA_PATH = "JSONFIles"
output = []

# Enhanced specialized prompts that support multiple keywords
SPECIALIZED_PROMPTS = [
    "Explain the relationship between {keywords} in computational complexity and their real-world implications.",
    "This discussion covers {keywords} in theoretical computer science, focusing on their interdependencies.",
    "Analyze {keywords} from a research paper and explore their significance in algorithm design.",
    "Provide a scientific context on {keywords} and how they interact in graph theory and complexity classes.",
    "Understanding the computational aspects of {keywords}, including their applications and limitations.",
    "Explore how {keywords} relate to P vs NP and their role in algorithmic efficiency.",
    "Discuss the impact of {keywords} on modern cryptographic systems and secure computation.",
    "How do {keywords} influence heuristic approaches in optimization and decision problems?",
    "Break down the theoretical and practical foundations of {keywords} in computational models.",
    "Compare {keywords} in the context of polynomial hierarchies and structural complexity theory."
]

for file in os.listdir(DATA_PATH):
    if file.endswith(".json"):
        with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for chunk in data:
                chunk_text = chunk.get("chunk_text", "").strip()
                metadata = chunk.get("metadata", {})
                keywords = metadata.get("keywords", [])
                num = random.randint(1,3)
                # Ensure multiple keywords are selected (2 or 3 if available)
                if num == 3:
                    if len(keywords) < 4 : 
                        print(file , chunk["id"])
                    selected_keywords = ", ".join(random.sample(keywords, 3))
                elif num == 2:
                    selected_keywords = ", ".join(keywords)
                elif num == 1:
                    selected_keywords = keywords[0]
                else:
                    selected_keywords = "computational problems and algorithms"

                if chunk_text:
                    prompt_template = random.choice(SPECIALIZED_PROMPTS)
                    prompt = prompt_template.format(keywords=selected_keywords)
                    output.append({
                        "prompt": prompt,
                        "response": chunk_text
                    })

with open(f"{DATA_PATH}/fine_tune_data.json", "w", encoding="utf-8") as out_file:
    json.dump(output, out_file, indent=2, ensure_ascii=False)
