import os
import subprocess
from chatbot import Chatbot


# Check if the database exists, if not, load it
db_path = "rnaseq.db"
if not os.path.exists(db_path):
    print("Database not found. Loading database...")
    subprocess.call("db_loader.py", shell=True)
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
