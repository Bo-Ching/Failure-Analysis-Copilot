import pandas as pd
import json

def generate_car_feedback_dataset(input_file, output_file):
    """
    專門針對車用資料集，從 Problem + Conversation + Report 建立回饋訓練資料。
    """
    df = pd.read_csv(input_file)
    records = []
    for _, row in df.iterrows():
        prompt = (
            f"問題：「{row['Problem']}」\n"
            f"RAG 回答：「{row['Conversation']}」\n"
            f"請你根據這個回答給出回饋。"
        )
        response = row["Report"]
        records.append({"prompt": prompt, "response": response})
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    print("🧪 測試 generate_car_feedback_dataset() ...")
    input_file = "../data/car_dataset/final_round_dev_set.csv"
    output_path = "test_generated_feedback_prompt.jsonl"

    df = pd.read_csv(input_file).head(5)
    temp_input = "temp_feedback_sample.csv"
    df.to_csv(temp_input, index=False, encoding="utf-8-sig")

    generate_car_feedback_dataset(temp_input, output_path)
    print(f"✅ 測試完成，結果儲存在 {output_path}")
