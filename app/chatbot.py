from flask import Flask, request, jsonify
from ner.intents import classify_query
import json
from handlers.bulk_rna_seq_handler import (
    get_most_expressed_gene
    # get_gene_expression,
    # plot_pca,
    # get_deg_list,
    # count_samples_by_group,
    # get_significant_genes
)


'''
use project_config.json to direct query to the correct omics technology,
the use intents.py to dipatch the query into a specific handler function
'''

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    user_input = request.json.get('query', '')
    if not user_input:
        return jsonify({'response': "Please provide a query."})

    # Classify the OMICS domain and step (intent), and the query type
    with open('project_config.json', 'r') as config_file:
        project_config = json.load(config_file)
    domain = project_config.get('omics_domain', 'unknown')
    intent = classify_query(user_input)

    # Debug print
    print(f"[DEBUG] Domain: {domain}, Intent: {intent}")

    # Dispatch the query based on its domain and intent
    if domain == "bulk_rna_seq":
        response = get_most_expressed_gene(user_input, step, query_type)
    elif domain == "wes":
        response = "WES handler is under development." 
    else:
        response = "Sorry, I cannot handle this query."
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
