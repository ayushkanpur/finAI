from transformers import AutoModelForCausalLM, AutoTokenizer

# download the LLM model and save locally in folder FinChat_XS_local

model_name = "oopere/FinChat-XS" # "lxyuan/distilgpt2-finetuned-finance"

# Download and save model locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save to a local directory (e.g., "finGPT_tiny_local/")
model.save_pretrained("FinChat_XS")
tokenizer.save_pretrained("FinChat_XS_local")

