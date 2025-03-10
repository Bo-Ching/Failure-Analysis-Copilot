import pandas as pd
import torch
from tqdm import tqdm
import json
from datetime import datetime
from models import load_llama3_model
from scripts.prompt_template import DATA_AUGMENTATION_PROMPT

def parse_multiple_enhancements(output):
    """解析 LLM 生成的多個增強版本"""
    enhancements = []
    current_enhancement = {}
    for line in output.split("\n"):
        if "Enhanced" in line:
            if current_enhancement:
                enhancements.append(current_enhancement)
                current_enhancement = {}
        elif ": " in line:
            key, value = line.split(": ", 1)
            current_enhancement[key.strip()] = value.strip() if value.strip() != "None" else None
    if current_enhancement:
        enhancements.append(current_enhancement)
    return enhancements

def data_augmentation(model, tokenizer, input_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = f"./data/augmented_8D_{timestamp}.json"

    df = pd.read_excel(input_path)

    combined_data = df.to_dict(orient="records")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Data"):
        if row['description'] and row['Corrective Action Verification']:
            texts = '\n'.join([f"{col}: {row[col]}" for col in ['description', 'WHY 1', 'WHY 2', 'WHY 3', 'WHY 4', 'WHY 5', 'Corrective Action Verification'] if col in row and not pd.isna(row[col])])
            
            prompt = DATA_AUGMENTATION_PROMPT.format(texts=texts)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=500)
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            enhancements = parse_multiple_enhancements(output)
            
            for idx, enhancement in enumerate(enhancements, start=1):
                enhancement['ISSUE NO.'] = f"{row['ISSUE NO.']}-Enhanced-{idx}"
                enhancement['Category'] = row['Category']
                enhancement['Occur1 Why'] = row['Occur1 Why']
                enhancement['Occur2 Why'] = row['Occur2 Why']
                enhancement['SFCS error code'] = row['SFCS error code']
                combined_data.append(enhancement)

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(combined_data, json_file, indent=4, ensure_ascii=False)

    print(f"Enhanced data saved to {output_json}")


if '__name__' == '__main__':
    model, tokenizer = load_llama3_model()
    INPUT_PATH = '../data/raw_docs/man_classification_and_8D.xlsx'
    data_augmentation(model=model,
                        tokenizer=tokenizer,
                        input_path=INPUT_PATH,)