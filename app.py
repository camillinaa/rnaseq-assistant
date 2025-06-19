import streamlit as st
import subprocess

# Set up the Streamlit interface
st.title("Simple Query Interface")

# Textbox for user input
query = st.text_input("Enter your query:")

# Output area
st.subheader("Output:")
if query:
    try:
        # Run main.py and pass the query as an argument
        result = subprocess.run(
            ["python3", "/Users/camilla.callierotti/omics-agent-rnaseq/main.py", query],
            capture_output=True,
            text=True
        )
        # Display the output from main.py
        st.write(result.stdout)
    except Exception as e:
        st.error(f"An error occurred: {e}")