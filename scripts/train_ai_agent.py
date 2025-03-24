from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from scripts.dataset import PromptDataset
import torch
import shutil
import os

# 訓練設定
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # ✅ LLaMA 3 8B instruct
data_path = "test_generated_question_prompt.jsonl"  # 可換成回饋資料
save_best_to = "models/llama3-ai-agent-best"

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 載入 base model + 加入 LoRA
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

# 建立資料集
dataset = PromptDataset(data_path, tokenizer)

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

if __name__ == "__main__":
    print("🚀 開始訓練 LLaMA 3 LoRA AI agent...")
    trainer.train()
    print("✅ 訓練完成！")

    # 儲存最佳模型到 models/
    best_model_dir = os.path.join(training_args.output_dir, "checkpoint-best")
    if os.path.exists(best_model_dir):
        print(f"📦 複製最佳模型到 {save_best_to}...")
        if os.path.exists(save_best_to):
            shutil.rmtree(save_best_to)
        shutil.copytree(best_model_dir, save_best_to)
        print("✅ 最佳模型已儲存！")
    else:
        print("⚠️ 未找到最佳模型 checkpoint，請確認訓練是否成功或正確啟用 load_best_model_at_end")