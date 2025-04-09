from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
    
intents = {
    "most_expressed_gene": "Which gene is most expressed in condition X?",
    "gene_expression_value": "What is the expression value of gene X in condition Y?",
    "de_gene_list": "Show me the list of differentially expressed genes between condition X and Y.",
    "sample_count_by_group": "How many replicates are there for each condition?",
    "plot_pca": "Show me a PCA plot of all samples."
    # Add more intents as needed
}

def classify_query(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    intent_embeddings = model.encode(list(intents.values()), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, intent_embeddings)
    best_match = list(intents.keys())[scores.argmax()]
    return best_match

# test

# # query1 = "How expressed is gene X in condition Y?" # PASS
# # query2 = "How expressed is TGF-beta in the control group?" # PASS
# # query3 = "Is CD8 downregulated in the knockdowns?" # FAIL
# # query4 = "To what extent is CD8 downregulated in the knockdowns?" # FAIL
# query5 = "To what extent is gene X downregulated in condition Y?" # PASS

# # answer1 = classify_query(query1)
# # answer2 = classify_query(query2)
# # answer3 = classify_query(query3)
# # answer4 = classify_query(query4)
# answer5 = classify_query(query5)

# # print(f"Query: {query1} => Classified Intent: {answer1}")
# # print(f"Query: {query2} => Classified Intent: {answer2}") 
# # print(f"Query: {query3} => Classified Intent: {answer3}")
# # print(f"Query: {query4} => Classified Intent: {answer4}")
# print(f"Query: {query5} => Classified Intent: {answer5}")