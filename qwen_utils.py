# qwen_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Qwen model once globally
print("Loading Qwen2-0.5B model locally...")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_qwen_reply(prompt: str, max_new_tokens: int = 200) -> str:
    """Generate response from Qwen2-0.5B"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,     # enable sampling for variety
        top_p=0.9,
        temperature=0.8
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()
