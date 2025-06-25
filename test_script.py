import os
from dotenv import load_dotenv
from chatbot import RNASeqChatbot  

load_dotenv()

# Retrieve secrets and config
api_key = os.getenv("MISTRAL_API_KEY")
db_uri = os.getenv("DB_URI")

# Create chatbot instance
chatbot = RNASeqChatbot(
    model_name="mistral-large-latest", 
    api_key=api_key,
    db_uri=db_uri,
    dialect="sqlite",
    top_k=10
)

# Test query
user_question = "Show me the most significantly differentially expressed genes"

# Generate SQL
sql = chatbot.sql_chain.invoke({
    "question": user_question,
    "table_info": chatbot.table_info,
    "dialect": chatbot.dialect,
    "top_k": chatbot.top_k
}).get("text", "")

print("Generated SQL:\n", sql)

# Execute query
df = chatbot.execute_sql(sql)
print("\nQuery results:\n", df)

# Decide plot type
plot_type = chatbot.plot_type(user_question, sql, df)
print("\nSuggested plot type:\n", plot_type)

# Generate plot code
plot_code = chatbot.plotly_chain.invoke({
    "question": user_question,
    "sql_query": sql,
    "data": df.to_dict(orient="records"),
    "columns": df.columns.tolist(),
    "plot_instructions": plot_type,
    "plot_type": plot_type
}).get("text", "")

print("\nGenerated plot code:\n", plot_code)

# Generate natural language response
response = chatbot.response_chain.invoke({
    "question": user_question,
    "data": df.to_dict(orient="records")
}).get("text", "")

print("\nChatbot response:\n", response)
