import streamlit as st
import pandas as pd
import torch
import fitz  # PyMuPDF for PDF parsing
import re
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# List of key financial terms to extract
FINANCIAL_TERMS = {
    "Total Revenue", "Net Income", "EBITDA", "Operating Income", 
    "Gross Profit", "Cost Of Revenue", "Tax Provision", "Total Expenses"
}

DATE_PATTERNS = [
    r"\b\d{4}\b",  # Match years like "2022", "2023"
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # Match dates like "12/31/2022"
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # Match dates like "2022-12-31"
    r"\bQ[1-4] \d{4}\b"  # Match "Q1 2023"
]

def load_model(model_name="FinChat_XS_local"):
    """Load the tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def extract_data(uploaded_file):
    """Extract and filter financial data from an Excel or PDF file."""
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
        filtered_data = filter_financial_data(df)
        return df, filtered_data

    elif file_type == "pdf":
        pdf_text = extract_text_from_pdf(uploaded_file)
        filtered_data = filter_financial_text(pdf_text)
        return None, filtered_data

    else:
        st.error("Unsupported file format. Please upload an Excel (.xlsx) or PDF file.")
        return None, None

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def filter_financial_data(df):
    """Filter key financial metrics from an Excel DataFrame."""
    extracted_info = {}
    for column in df.columns:
        for term in FINANCIAL_TERMS:
            if term.lower() in column.lower():
                extracted_info[term] = df[column].dropna().tolist()
    return extracted_info

def filter_financial_text(text):
    """Extract relevant financial terms and values from unstructured text (PDF)."""
    extracted_info = {}
    
    # Extract dates
    dates_found = []
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text)
        dates_found.extend(matches)
    
    # Extract key financial terms
    for term in FINANCIAL_TERMS:
        pattern = rf"\b{re.escape(term)}\b.*?([-+]?\d*\.\d+|\d+)"  # Extract key + numeric value
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            extracted_info[term] = [float(match) for match in matches]

    if dates_found:
        extracted_info["Report Dates"] = list(set(dates_found))

    return extracted_info

def generate_response(model, tokenizer, financial_data, user_query):
    """Generate a response from the financial AI model based on user query."""
    if not financial_data:
        financial_data = "none. answer with non-financial data."

    context_text = "\n".join([f"{key}: {', '.join(map(str, values))}" for key, values in financial_data.items()])

    full_prompt = (
        "Answer the user's question based only on the provided financial data. "
        "- Extracted Financial Data:\n"
        f"{context_text[:1500]}...\n\n"
        "-User Question:\n"
        f"{user_query}\n\n"
        "Answer concisely with meaningful insights:"
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=300,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        repetition_penalty=1
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def visualize_trends(financial_data):
    """Generate graphs based on extracted financial data trends."""
    st.write("### 📈 Financial Data Trends")

    # Filter only numerical data
    trend_data = {key: values for key, values in financial_data.items() if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values)}

    if not trend_data:
        st.warning("No valid financial trend data found for visualization.")
        return

    df = pd.DataFrame(trend_data)

    # Plot trends
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, markers=True, dashes=False)
    
    plt.xlabel("Time (Sequential Entries)")
    plt.ylabel("Financial Value")
    plt.title("Financial Trends Over Time")
    plt.legend(title="Financial Metrics")
    plt.xticks(rotation=45)

    st.pyplot(fig)




def main():
    st.set_page_config(page_title="Financial AI Chat", page_icon=":moneybag:")
    st.title("🦜 Financial AI Chat Assistant")
    st.markdown("Upload a financial report (.xlsx or .pdf) and chat with AI about it.")

    tokenizer, model = load_model()
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'pdf'])

    if uploaded_file:
        df, filtered_data = extract_data(uploaded_file)

        if df is not None:
            st.write("### Extracted Data (Excel)")
            st.dataframe(df)
        elif filtered_data:
            st.write("### Extracted Financial Metrics")
            st.json(filtered_data)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.text_input("Ask a question about the report:")

        if user_query:
            with st.spinner("Processing..."):
                response = generate_response(model, tokenizer, filtered_data, user_query)
                st.session_state.chat_history.insert(0, ("**You:** " + user_query, "**AI:** " + response))

        if st.session_state.chat_history:
            st.write("### Chat History")
            for entry in st.session_state.chat_history:
                st.markdown(entry[1])
                st.markdown("---")

        # 📊 Visualization Button
        if st.button("📊 Visualize"):
            visualize_trends(filtered_data)

if __name__ == "__main__":
    main()
