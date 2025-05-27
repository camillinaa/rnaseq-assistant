import streamlit as st
import pandas as pd
from main import get_available_subsets_and_comparisons, load_analysis_table
from deseq2_handler import run_query
import pandasai as pai
from dotenv import load_dotenv
import os

# Set up the API key for pandasai
load_dotenv()
api_key = os.getenv("SECRET_API_KEY")
pai.api_key.set(api_key)

# Get selections
subset_comparisons = get_available_subsets_and_comparisons()
analysis_types = ["deseq2", "ora", "gsea"]

# Title
st.set_page_config(page_title="Analysis Results Chatbot", page_icon=":bar_chart:")
st.title("Analysis Results Chatbot")
st.subheader("Human Technopole National Facility for Data Handling and Analysis")
st.write("Welcome to the Analysis Results Chatbot. This chatbot will help you navigate through the analysis results. Please select the type of analysis and the corresponding comparisons to query your data.")

# First dropdown: analysis type
selected_analysis = st.selectbox("Select analysis type", analysis_types)

# Second dropdown: subset
selected_subset = st.selectbox("Select sample subset", list(subset_comparisons.keys()))

# Third dropdown: comparison (based on chosen subset)
selected_comparison = st.selectbox("Select comparison", subset_comparisons[selected_subset])

# Get filepath

_, filepath = load_analysis_table(selected_subset, selected_comparison, selected_analysis)

# Button to run query or load file
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None

if st.button("Confirm and Load Analysis Table"):
    st.write(f"You selected subset: `{selected_subset}`, comparison: `{selected_comparison}`, analysis: `{selected_analysis}`")

    # Here, call function from main.py to show the table
    df, _ = load_analysis_table(selected_subset, selected_comparison, selected_analysis)
    st.session_state.loaded_data = df

# Display the loaded data if available
if st.session_state.loaded_data is not None:
    st.dataframe(st.session_state.loaded_data)

user_query = st.text_input("Ask a question about the analysis table")
if st.button("Run Query"):
    result = run_query(user_query, filepath)
    st.write("Query Result:")
    st.write(result)

