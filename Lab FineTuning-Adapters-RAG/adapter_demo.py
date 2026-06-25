from transformers import AutoModelForCausalLM, AutoTokenizer

print("\n===== ADAPTERS / LORA DEMO =====\n")

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Base model loaded:", model_name)

print("\n👉 In real LoRA/Adapters:")
print("- Only small layers are trained")
print("- Base model weights are frozen")
print("- Much cheaper than full fine-tuning")

print("\n👉 This is a conceptual demo (no training executed).")