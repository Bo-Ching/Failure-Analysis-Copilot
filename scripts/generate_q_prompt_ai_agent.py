# 📄 檔案位置: scripts/generate_question_prompt.py

import pandas as pd
import json

def generate_car_question_dataset(input_file, output_file):
    """
    專門針對車用資料集，將問題轉為 LLaMA instruct 格式的提問任務資料。
    """
    df = pd.read_csv(input_file)
    records = []
    for _, row in df.iterrows():
        prompt = f"這是一台 {row['Brand']} {row['Collection']}，顧客描述問題為：「{row['Problem']}」。請模擬顧客會怎麼問技師？"
        response = row["Problem"]  # 可以用 Problem 做為目標問法的近似
        records.append({"prompt": prompt, "response": response})
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    print("🧪 測試 generate_car_question_dataset() ...")
    input_file = "../data/car_dataset/final_round_dev_set.csv"
    output_path = "test_generated_question_prompt.jsonl"

    # 只取前 5 筆測試
    df = pd.read_csv(input_file).head(5)
    temp_input = "temp_question_sample.csv"
    df.to_csv(temp_input, index=False, encoding="utf-8-sig")

    generate_car_question_dataset(temp_input, output_path)
    print(f"✅ 測試完成，結果儲存在 {output_path}")
