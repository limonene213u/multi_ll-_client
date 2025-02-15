# llm_inference.py
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルをロード
MODEL_NAME = "rinna/nekomata-7b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = sys.argv[1]
    print(generate_response(prompt))
