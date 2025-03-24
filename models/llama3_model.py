from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"

"""load Llama-3-8B-Instruct model and tokenizer"""
def load_llama3_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.float16,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer