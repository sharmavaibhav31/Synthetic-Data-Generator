import pandas as pd
import random
from faker import Faker
import os
import json
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re

# Initialize
load_dotenv()
fake = Faker()

# Load model & tokenizer
model_name = os.getenv("MODEL_NAME", "google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def prompt_model(prompt: str, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_example_mapping(response: str):
    """
    Extract JSON object from model output string.
    """
    print("\n=== RAW MODEL OUTPUT ===\n")
    print(response)

    try:
        # Try to extract first JSON-looking object
        json_match = re.search(r'\{[\s\S]*?\}', response)

        if not json_match:
            raise ValueError("No JSON object found in model output.")

        raw_json = json_match.group()

        # Clean up to help parsing
        cleaned = raw_json.replace("'", '"')
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)  # Remove trailing commas
        cleaned = re.sub(r"[“”]", '"', cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Parse
        result = json.loads(cleaned)

        if not isinstance(result, dict):
            raise ValueError("Parsed object is not a dictionary.")

        return result

    except Exception as e:
        raise ValueError("Failed to parse model output: " + str(e))

def infer_generator(example_value):
    if isinstance(example_value, int):
        return lambda: random.randint(18, 90)
    elif isinstance(example_value, float):
        return lambda: round(random.uniform(1000, 9999), 2)
    elif isinstance(example_value, str):
        ev = example_value.lower()
        if "@" in ev:
            return fake.email
        elif ev in ["male", "female", "other"]:
            return lambda: random.choice(["Male", "Female", "Other"])
        elif ev in ["o+", "a-", "b+", "ab-", "ab+", "a+", "b-", "o-"]:
            return lambda: random.choice(["O+", "A+", "B+", "AB+", "O−", "A−", "B−", "AB−"])
        elif re.search(r"\d{4}-\d{2}-\d{2}", ev):
            return fake.date
        elif len(example_value.split()) >= 2:
            return fake.name
        else:
            return lambda: fake.word()
    else:
        return lambda: "N/A"

def generate_synthetic_data(topic, fields, records, file_format="excel"):
    # Step 1: Model prompt
    prompt = (
        f"Generate a JSON object with exactly {fields} key-value pairs related to the topic '{topic}'. "
        f"The keys must be relevant and realistic column names and values must be example values. "
        f"Respond ONLY with the JSON object. No explanations, no extra text."
    )

    response = prompt_model(prompt)

    try:
        example_dict = parse_example_mapping(response)
    except Exception as e:
        raise Exception("Failed to parse model output: " + str(e))

    if len(example_dict) != fields:
        raise Exception(f"Model returned {len(example_dict)} fields, expected {fields}.")

    # Step 2: Setup generators
    generators = {field: infer_generator(example) for field, example in example_dict.items()}

    # Step 3: Generate data
    data = []
    for _ in range(records):
        row = [gen() for gen in generators.values()]
        data.append(row)

    df = pd.DataFrame(data, columns=list(generators.keys()))

    # Step 4: Save
    filename = "generated_data.xlsx" if file_format == "excel" else "generated_data.csv"
    if file_format == "excel":
        df.to_excel(filename, index=False)
    else:
        df.to_csv(filename, index=False)

    return filename
