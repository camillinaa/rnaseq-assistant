import os
# import streamlit as st
# import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util

subset_comparisons = {}
subsets = []

def get_available_subsets_and_comparisons():
    """
    Get available subsets and comparisons for this project from the data directory (analyses are hardcoded).
    """
    for root, dirnames, filenames in os.walk("data", topdown=False):
        for name in dirnames:
            if name.startswith("dea_"):
                subset_name = name[4:].replace('_', ' ').capitalize()
                subsets.append(subset_name)
                for subset in subsets:
                    subset_path = os.path.join("data", f'dea_{subset.lower().replace(" ", "_")}')
                    if os.path.isdir(subset_path):
                        subset_comparisons[subset] = []
                        for folder in os.listdir(subset_path):
                            folder_path = os.path.join(subset_path, folder)
                            if os.path.isdir(folder_path) and folder.startswith('dea_'):
                                comparison_name = folder[4:].replace('_', ' ').capitalize()
                                subset_comparisons[subset].append(comparison_name)
    return subset_comparisons

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("config/intents.json", "r") as f:
    intents = json.load(f)

def classify_query(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    intent_embeddings = model.encode(list(intents.values()), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, intent_embeddings)
    best_match = list(intents.keys())[scores.argmax()]
    return best_match


# def load_analysis_table(selected_analysis, selected_subset, selected_comparison):
#     """
#     Load the table to query based on the analysis, subset, and comparison selected by the user on streamlit app.
#     """
    
#     if selected_analysis in ["counts matrix", "PCA", "MDS"]:
#         selected_subset = None
#         selected_comparison = None
#     else:
#         subset = selected_subset.lower().replace(' ', '_')
#         comparison = selected_comparison.lower().replace(' ', '_')
    
#     root = "data"

#     if selected_analysis == "counts matrix":
#         file_path = os.path.join(root, "normalization/cpm.txt")
#         df = pd.read_csv(file_path, sep="\t")
#     elif selected_analysis == "PCA":    
#         file_path = os.path.join(root, "dim_reduction/PCA_scores.txt")
#         df = pd.read_csv(file_path, sep="\t")
#     elif selected_analysis == "MDS":
#         file_path = os.path.join(root, "dim_reduction/MDS_scores.txt")
#         df = pd.read_csv(file_path, sep="\t")
#     elif selected_analysis == "deseq2":
#         file_path = os.path.join(root, f"dea_{subset}", f"dea_{comparison}", f"deseq2_toptable.{comparison}.txt")
#         with open(file_path, 'r') as file:
#             df = pd.read_csv(file, sep="\t")
#     elif selected_analysis == "ora":
#         file_path = os.path.join(root, f"dea_{subset}", f"dea_{comparison}", f"ora_CP.{comparison}.all.xlsx")
#         df = pd.concat(pd.read_excel(file_path, sheet_name=None), ignore_index=True) # check
#     elif selected_analysis == "gsea":
#         file_path = os.path.join(root, f"dea_{subset}", f"dea_{comparison}", f"gsea.{comparison}.xlsx")
#         df = pd.concat(pd.read_excel(file_path, sheet_name=None), ignore_index=True) # check
#     else: 
#         raise ValueError(f"Unknown analysis type: {selected_analysis}")
    
#     return df, file_path
