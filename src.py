import streamlit as st
import pandas as pd
import torch
import fitz  # PyMuPDF for PDF parsing
import re
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

FINANCIAL_TERMS = {
    # General Financials
    "Total Revenue", "Operating Revenue", "Gross Profit", "Cost of Revenue", "Operating Expenses", 
    "EBITDA", "Net Income from Continuing Operations", "Normalized EBITDA", "Total Unusual Items", 
    "Tax Rate for Calculations", "Tax Effect of Unusual Items", "Reconciled Cost of Revenue",
    "Reconciled Depreciation", "Research and Development (R&D) Expense", "Selling, General, and Administrative (SG&A) Expenses", 
    "General and Administrative Expense", "Selling and Marketing Expense", "Rent and Landing Fees",
    
    # Balance Sheet
    "Total Assets", "Total Liabilities", "Total Equity", "Cash and Equivalents", "Accounts Receivable", 
    "Inventory", "Property, Plant & Equipment (PP&E)", "Total Debt", "Current Liabilities", "Current Assets", 
    "Long-Term Debt", "Short-Term Debt", "Retained Earnings", "Shareholders’ Equity", 
    
    # Cash Flow Statement
    "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Capital Expenditures", "Free Cash Flow", 
    "Depreciation & Amortization", "Stock-Based Compensation", "Change in Working Capital",
    
    # Key Statistics
    "Earnings Per Share (EPS)", "Price-to-Earnings Ratio (P/E)", "Return on Assets (ROA)", "Return on Equity (ROE)",
    "Debt-to-Equity Ratio", "Current Ratio", "Quick Ratio", "Dividend Yield"
}

DATE_PATTERNS = [
    r"\b\d{4}\b",  
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",  
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",  
    r"\bQ[1-4] \d{4}\b"  
]

def load_model(model_name="FinChat_XS_local"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def extract_data(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
        financial_data = filter_financial_data(df)
        general_text = extract_general_text(df)
        return df, financial_data, general_text

    elif file_type in ["pdf", "txt"]:
        pdf_text = extract_text_from_pdf(uploaded_file, file_type)
        financial_data = filter_financial_text(pdf_text)
        general_text = extract_general_text_from_text(pdf_text)
        return None, financial_data, general_text

    else:
        st.error("Unsupported file format. Please upload an Excel (.xlsx) or PDF file.")
        return None, None, None

def extract_text_from_pdf(pdf_file, file_type):
    """Extract text from PDF or TXT file."""
    if file_type == "pdf":
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
    elif file_type == "txt":
        text = pdf_file.read().decode("utf-8")
    return text

def filter_financial_data(df):
    extracted_info = {}
    for column in df.columns:
        for term in FINANCIAL_TERMS:
            if term.lower() in column.lower():
                extracted_info[term] = df[column].dropna().tolist()
    return extracted_info

def filter_financial_text(text):
    extracted_info = {}
    dates_found = []
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text)
        dates_found.extend(matches)
    
    for term in FINANCIAL_TERMS:
        pattern = rf"\b{re.escape(term)}\b.*?([-+]?\d*\.\d+|\d+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            extracted_info[term] = [float(match) for match in matches]

    if dates_found:
        extracted_info["Report Dates"] = list(set(dates_found))

    return extracted_info

def extract_general_text(df):
    """Extract non-financial text from Excel (assumes financial terms are structured)."""
    text_blocks = []
    for column in df.columns:
        if all(isinstance(val, str) for val in df[column].dropna()):
            text_blocks.append(" ".join(df[column].dropna().astype(str)))
    
    return " ".join(text_blocks) if text_blocks else ""

def extract_general_text_from_text(text):
    """Extract non-financial text from a document."""
    lines = text.split("\n")
    non_financial_lines = [line for line in lines if not any(term in line for term in FINANCIAL_TERMS)]
    return "\n".join(non_financial_lines)

def generate_response(model, tokenizer, financial_data, general_text, user_query, include_story):
    context_text = ""

    if include_story and general_text:
        context_text += f"General Story:\n{general_text[:10000]}...\n\n"

    if financial_data:
        financial_context = "\n".join([f"{key}: {', '.join(map(str, values))}" for key, values in financial_data.items()])
        context_text += f"Extracted Financial Data:\n{financial_context[:2000]}...\n\n"
    
    if not context_text:
        context_text = "No relevant financial or general text found. Answer generically."

    full_prompt = (
        "Analyze and respond based on the given context:\n"
        f"{context_text}\n\n"
        "-User Question:\n"
        f"{user_query}\n\n"
        "**Provide a meaningful response with insights:**"
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
    st.write("### 📈 Financial Data Trends")

    trend_data = {key: values for key, values in financial_data.items() if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values)}

    if not trend_data:
        st.warning("No valid financial trend data found for visualization.")
        return

    df = pd.DataFrame(trend_data)
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
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'pdf', 'txt'])
    include_story = st.checkbox("Analyze Story (Include General Text)")

    if uploaded_file:
        df, financial_data, general_text = extract_data(uploaded_file)

        if df is not None:
            st.write("### Extracted Data (Excel)")
            st.dataframe(df)
        elif financial_data:
            st.write("### Extracted Financial Metrics")
            st.json(financial_data)

        if include_story and general_text:
            st.write("### Extracted General Story")
            st.text_area("General Context", value=general_text[:2000], height=200)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.text_input("Ask a question about the report:")

        if user_query:
            with st.spinner("Processing..."):
                response = generate_response(model, tokenizer, financial_data, general_text, user_query, include_story)
                st.session_state.chat_history.insert(0, ("**You:** " + user_query, "**AI:** " + response))

        if st.session_state.chat_history:
            st.write("### Chat History")
            for entry in st.session_state.chat_history:
                st.markdown(entry[1])
                st.markdown("---")

        if st.button("📊 Visualize Trends"):
            visualize_trends(financial_data)

if __name__ == "__main__":
    main()
