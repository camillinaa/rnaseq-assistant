import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("/Users/camilla.callierotti/omics-agent/config/bulk_rna_seq_intents.json", "r") as f:
    intents = json.load(f)

def classify_query(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    intent_embeddings = model.encode(list(intents.values()), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, intent_embeddings)
    best_match = list(intents.keys())[scores.argmax()]
    return best_match


# tests

# query1 = "How expressed is gene X in condition Y?" # PASS
# query2 = "Is transcript TGF-beta upregulated in the treated group?" # PASS
# query3 = "Is CD8 downregulated in the knockdowns?" # FAIL - doesn't know that CD8 is a gene and not a pathway
# query4 = " is CD8 gene downregulated in the knockdowns condition?" # PASS
# query5 = "To what extent is gene X downregulated in condition Y?" # PASS
# query5 = "What enriched pathways are there?" # PASS

# answer1 = classify_query(query1)
# answer2 = classify_query(query2)
# answer3 = classify_query(query3)
# answer4 = classify_query(query4)
# answer5 = classify_query(query5)

# print(f"Query: {query1} => Classified Intent: {answer1}")
# print(f"Query: {query2} => Classified Intent: {answer2}") 
# print(f"Query: {query3} => Classified Intent: {answer3}")
# print(f"Query: {query4} => Classified Intent: {answer4}")
# print(f"Query: {query5} => Classified Intent: {answer5}")