# 📄 檔案位置: scripts/finetune_car_agent.py

from scripts import (
    convert_file_to_traditional,
    generate_car_question_dataset,
    PromptDataset
)
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
from models.llama3_model import load_llama3_model
import torch
import os
import shutil
import random
import json
import matplotlib.pyplot as plt


def finetune_car_agent():
    print("🚗 開始微調汽車問題 AI agent...")

    # === 1. 簡體轉繁體 ===
    input_csv = "data/car_dataset/final_round_dev_set.csv"
    trad_csv = "data/car_dataset/final_round_dev_set_traditional.csv"
    convert_file_to_traditional(
        input_path=input_csv,
        output_path=trad_csv,
        text_columns=["Brand", "Collection", "Problem", "Conversation", "Report"]
    )

    # === 2. 產生微調資料 ===
    jsonl_path = "data/processed/ai_agent_question.jsonl"
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    generate_car_question_dataset(trad_csv, jsonl_path)

    # === 3. 載入模型與 Tokenizer（封裝版） ===
    base_model, tokenizer = load_llama3_model()

    # === 4. 加入 LoRA 設定 ===
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)

    # === 5. 建立 Dataset 與驗證資料集 ===
    with open(jsonl_path, encoding='utf-8') as f:
        lines = [json.loads(line) for line in f.readlines()]
    random.shuffle(lines)
    split = int(len(lines) * 0.9)
    train_lines = lines[:split]
    eval_lines = lines[split:]

    train_path = "data/processed/ai_agent_question_train.jsonl"
    eval_path = "data/processed/ai_agent_question_eval.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for l in train_lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    with open(eval_path, 'w', encoding='utf-8') as f:
        for l in eval_lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    train_dataset = PromptDataset(train_path, tokenizer)
    eval_dataset = PromptDataset(eval_path, tokenizer)

    # === 6. 訓練模型 ===
    training_args = TrainingArguments(
        output_dir="outputs/llama3-ai-agent-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        bf16=False,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        eval_steps=None,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        logging_dir="logs"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    training_result = trainer.train()

    # === 7. 儲存最佳模型 ===
    best_model_path = os.path.join(training_args.output_dir, "checkpoint-best")
    final_save_path = "models/llama3-ai-agent-best"
    if os.path.exists(best_model_path):
        if os.path.exists(final_save_path):
            shutil.rmtree(final_save_path)
        shutil.copytree(best_model_path, final_save_path)
        print("✅ 最佳模型已儲存於:", final_save_path)
    else:
        print("⚠️ 未找到最佳模型 checkpoint")

    # === 8. 繪製 loss 圖表 ===
    logs = training_result.metrics
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        history = trainer.state.log_history
        train_losses = [x["loss"] for x in history if "loss" in x]
        eval_losses = [x["eval_loss"] for x in history if "eval_loss" in x]
        steps = list(range(1, len(eval_losses) + 1))

        plt.figure()
        plt.plot(steps, eval_losses, label="Eval Loss", marker='o')
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Eval Loss")
        plt.legend()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/loss_curve.png")
        print("📉 Loss 曲線已儲存為 outputs/loss_curve.png")