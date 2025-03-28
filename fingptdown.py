from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "lxyuan/distilgpt2-finetuned-finance"

# Download and save model locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save to a local directory (e.g., "finGPT_tiny_local/")
model.save_pretrained("distilgpt2_finetuned_finance_local")
tokenizer.save_pretrained("distilgpt2_finetuned_finance_local")

