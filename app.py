import streamlit as st
import pandas as pd
from main import get_available_subsets_and_comparisons, load_analysis_table
from handler import run_counts_query, run_pca_mds_query, run_deseq2_query


# Get selections
subset_comparisons = get_available_subsets_and_comparisons()
analysis_types = ["counts matrix", "PCA", "MDS", "deseq2", "ora", "gsea"]

# Title
st.set_page_config(page_title="Analysis Results Chatbot", page_icon=":bar_chart:")
st.title("Analysis Results Chatbot")
st.subheader("Human Technopole National Facility for Data Handling and Analysis")
st.write("Welcome to the Analysis Results Chatbot. This chatbot will help you navigate through the analysis results. Please select the type of analysis and the corresponding comparisons to query your data.")

# First dropdown: analysis type
selected_analysis = st.selectbox("Select analysis type", analysis_types)

# Show subset and comparison dropdowns only for specific analysis types
if selected_analysis not in ["counts matrix", "PCA", "MDS"]:
    selected_subset = st.selectbox("Select sample subset", list(subset_comparisons.keys()))
    selected_comparison = st.selectbox("Select comparison", subset_comparisons[selected_subset])
else:
    selected_subset = None
    selected_comparison = None

# Get filepath
_, filepath = load_analysis_table(selected_analysis, selected_subset, selected_comparison)

# Button to run query or load file
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None

if st.button("Confirm and Load Analysis Table"):
    st.write(f"You selected subset: `{selected_subset}`, comparison: `{selected_comparison}`, analysis: `{selected_analysis}`")

    # Here, call function from main.py to show the table
    df, _ = load_analysis_table(selected_analysis, selected_subset, selected_comparison)
    st.session_state.loaded_data = df

# Display the loaded data if available
if st.session_state.loaded_data is not None:
    st.dataframe(st.session_state.loaded_data)

user_query = st.text_input("Ask a question about the analysis table")
if st.button("Run Query"):
    if selected_analysis == "counts matrix":
        result = run_counts_query(user_query, filepath)
    if selected_analysis == "PCA" or selected_analysis == "MDS":
        result = run_pca_mds_query(user_query, filepath)
    elif selected_analysis == "deseq2":
        result = run_deseq2_query(user_query, filepath)
    else:
        st.error("This analysis type is not supported for querying yet.")
    st.write("Query Result:")
    st.write(result)

