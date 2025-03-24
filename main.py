from scripts import (
    convert_file_to_traditional,
    generate_car_question_dataset,
    generate_car_feedback_dataset,
    PromptDataset
)

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import os
import shutil
import torch

if __name__ == "__main__":
    print("🚀 啟動 LLaMA 3 AI agent 微調流程...")

    # === 1. 將簡體轉為繁體 ===
    input_csv = "data/car_dataset/final_round_dev_set.csv"
    trad_csv = "data/car_dataset/final_round_dev_set_traditional.csv"
    convert_file_to_traditional(
        input_path=input_csv,
        output_path=trad_csv,
        text_columns=["Brand", "Collection", "Problem", "Conversation", "Report"]
    )
    print("✅ 繁體轉換完成")

    # === 2. 產生微調資料（可改為 generate_car_feedback_dataset） ===
    jsonl_path = "data/processed/ai_agent_question.jsonl"
    generate_car_question_dataset(trad_csv, jsonl_path)
    print("✅ 微調資料產出完成")

    # === 3. 準備模型與 Tokenizer ===
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)

    # === 4. 建立 Dataset ===
    dataset = PromptDataset(jsonl_path, tokenizer)

    # === 5. 訓練參數與 Trainer 設定 ===
    training_args = TrainingArguments(
        output_dir="outputs/llama3-ai-agent-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args
    )

    print("🧠 開始訓練模型...")
    trainer.train()
    print("✅ 訓練完成")

    # === 6. 儲存最佳模型 ===
    best_model_path = os.path.join(training_args.output_dir, "checkpoint-best")
    final_save_path = "models/llama3-ai-agent-best"
    if os.path.exists(best_model_path):
        print(f"💾 儲存最佳模型到 {final_save_path}...")
        if os.path.exists(final_save_path):
            shutil.rmtree(final_save_path)
        shutil.copytree(best_model_path, final_save_path)
        print("✅ 最佳模型已儲存")
    else:
        print("⚠️ 未找到最佳模型 checkpoint")
