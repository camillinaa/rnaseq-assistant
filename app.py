import streamlit as st
from chatbot import RNASeqChatbot
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Instantiate chatbot once at the top (Streamlit will cache it)
def load_chatbot():
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("MODEL_NAME") # "open-mistral-7b" 
    db_uri = os.getenv("DB_URI")
    dialect = "sqlite"
    top_k = 100
    return RNASeqChatbot(model_name, api_key, db_uri, dialect, top_k)

chatbot = load_chatbot()

st.title("🧬 OMICS-query 🧬")
st.markdown("Ask any question about your RNA-seq data.")

user_query = st.text_input("Enter your question:")

if user_query:
    with st.spinner("Processing your query..."):
        try:
            # Step 1: SQL query
            sql_chain_result = chatbot.sql_chain.invoke({
                "question": user_query,
                "table_info": chatbot.table_info,
                "dialect": chatbot.dialect,
                "top_k": chatbot.top_k
            })
            parsed_sql = chatbot._extract_structured_output(sql_chain_result, chatbot.sql_output_parser)
            sql_query = parsed_sql["sql_query"]

            st.subheader("Generated SQL Query")
            st.code(sql_query, language="sql")

            # Step 2: Run SQL
            df = chatbot.execute_sql(sql_query)
            if df.empty:
                st.warning("No data returned by this query.")
            else:
                st.subheader("Query Results")
                st.dataframe(df)

                # Step 3: Determine plot
                plot_type = chatbot.determine_plot_type(user_query, sql_query, df)

                if plot_type != "no visualization":
                    st.subheader(f"Generated Visualization:")
                    fig = chatbot.generate_plot(user_query, df, sql_query, plot_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        if st.download_button("📥 Download as HTML", fig.to_html(), file_name="query_result.html"):
                            st.success("Download started.")

                # Step 4: Generate response
                response = chatbot.generate_response(user_query, df)
                st.subheader("Analysis")
                st.markdown(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
