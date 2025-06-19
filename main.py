import os
from db_loader import RNASeqDBLoader
from chatbot import Chatbot


# Check if the database exists, if not, load it
db_path = "rnaseq.db"
if not os.path.exists(db_path):
    print("Database not found. Loading database...")
    RNASeqDBLoader(db_path=db_path, base_dir="data").upload_files_to_db()
    print("Database loaded successfully.")
else:
    print("Database already loaded.")

# Get user input and process it using the chatbot
user_query = input("Enter your query: ")
bot = Chatbot(
    db_uri="sqlite:///rnaseq.db",
    model_name="mistral-large-latest",
    model_provider="mistralai"
)
response = bot.run_query(user_query)
print("Response:", response)
