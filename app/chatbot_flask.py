import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from flask import Flask, request, jsonify
import requests
from ner.intents import classify_query
from handlers.bulk_rna_seq_handler import (
    # get_most_expressed_gene,
    # get_gene_expression,
    # plot_pca,
    # get_deg_list,
    count_samples_by_group
    # get_significant_genes
)

'''
use project_config.json to direct query to the correct omics technology,
the use intents.py along with the intent_to_handler.json 
to dipatch the query to a specific handler function
'''

# Test configuration (will be replaced by project config for domain and path)
with open(os.path.join(os.path.dirname(__file__), '../config/test_config.json'), 'r') as config_file:
    test_config = json.load(config_file)
user_input = test_config.get("user_input", "")
intent = test_config.get("intent", "")
omics_domain = test_config.get("omics_domain", "")
condition = test_config.get("condition", "")
proj_path = test_config.get("proj_path", "")

print(f"[DEBUG] Test Config: {user_input}, {intent}, {omics_domain}, {condition}, {proj_path}")


app = Flask(__name__)

# Define route for simple health check
@app.route("/", methods=["GET"])
def home():
    return "OMICS Chatbot is up and running!"

# Define chat endpoint
@app.route("/query", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("query", "").strip()

    if not user_input:
        return jsonify({"response": "Please provide a valid query."})

    # Step 1: Classify intent
    intent = classify_query(user_input)

    # Step 2: Get domain info and project metadata
    domain = test_config.get("omics_domain", "unknown")
    proj_path = test_config.get("proj_path", "")
    condition = test_config.get("condition", "")

    print(f"[DEBUG] Input: {user_input}")
    print(f"[DEBUG] Domain: {domain}, Intent: {intent}")

    # Step 3: Dispatch based on domain and intent
    if domain == "bulk_rna_seq":
        if intent == "sample_count_by_group":
            response = count_samples_by_group(proj_path, condition)
        else:
            response = f"No handler implemented for intent: {intent}"
    elif domain == "wes":
        response = "WES handler is under development."
    else:
        response = "Sorry, I cannot handle this query."

    print(f"[DEBUG] Response: {response}")
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)