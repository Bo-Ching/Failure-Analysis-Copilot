import pandas as pd
import opencc

def convert_file_to_traditional(input_path, output_path, text_columns):
    """
    將指定欄位從簡體轉繁體，並儲存轉換後的 CSV 檔案。
    """
    converter = opencc.OpenCC('s2t')
    df = pd.read_csv(input_path)
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(converter.convert)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    print("🔁 測試簡繁轉換功能...")

    input_path = "../data/car_dataset/final_round_dev_set.csv"
    output_path = "test_output_traditional.csv"
    text_columns = ["Brand", "Collection", "Problem", "Conversation", "Report"]

    # 讀取資料前 5 筆作為測試
    df = pd.read_csv(input_path).head(5)
    temp_input_path = "test_input_subset.csv"
    df.to_csv(temp_input_path, index=False, encoding='utf-8-sig')

    convert_file_to_traditional(temp_input_path, output_path, text_columns)
    print(f"✅ 測試完成，結果儲存於 {output_path}")