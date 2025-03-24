from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json

class PromptDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = [json.loads(line) for line in open(data_path, encoding='utf-8')]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt = self.samples[idx]['prompt']
        response = self.samples[idx]['response']
        input_text = f"<s>[INST] {prompt} [/INST] {response}</s>"

        encodings = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        encodings = {k: v.squeeze(0) for k, v in encodings.items()}
        encodings["labels"] = encodings["input_ids"].clone()
        return encodings

if __name__ == "__main__":
    print("🧪 測試 PromptDataset 建立與取樣...")
    test_path = "test_generated_question_prompt.jsonl"  # 或改成 feedback 的測試檔
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = PromptDataset(test_path, tokenizer)
    sample = dataset[0]

    print("✅ 測試成功：")
    print(f"Prompt tokens: {sample['input_ids'][:20]}")
    print(f"Label tokens:  {sample['labels'][:20]}")
