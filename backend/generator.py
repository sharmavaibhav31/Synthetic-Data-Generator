import pandas as pd
from faker import Faker
import random
from transformers import pipeline

fake = Faker()

# Load the FLAN-T5 model
text2text = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_field_names(topic, num_fields):
    prompt = f"List {num_fields} realistic column headers for a dataset about {topic}."
    response = text2text(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']

    # Try splitting by commas or newlines
    field_list = [f.strip().strip('–-:') for f in response.replace("\n", ",").split(",") if f.strip()]
    
    # Use only first N field names
    if len(field_list) < num_fields:
        print("Warning: Not enough fields parsed. Using default names.")
        field_list = [f"Field_{i+1}" for i in range(num_fields)]
    
    return field_list[:num_fields]

def generate_synthetic_data(topic, fields, records, file_format="excel"):
    column_names = generate_field_names(topic, fields)

    faker_methods = [
        fake.name, fake.address, fake.date_of_birth, fake.job, fake.company,
        fake.email, fake.phone_number, fake.text, fake.country, fake.city
    ]

    data = []
    for _ in range(records):
        row = [random.choice(faker_methods)() for _ in range(fields)]
        data.append(row)

    df = pd.DataFrame(data, columns=column_names)

    filename = "generated_data.xlsx" if file_format == "excel" else "generated_data.csv"

    if file_format == "excel":
        df.to_excel(filename, index=False)
    else:
        df.to_csv(filename, index=False)

    return filename
