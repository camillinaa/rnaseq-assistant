import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loaders.bulk_rna_seq_loader import (
    load_metadata
    # load_counts_matrix,
    # load_dge_results,
    # load_enrichr_results,
    # load_pca,
    # load_mds
)
from ner.attributes import create_metadata_ner
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re

### 1. METADATA QUERIES ###

def find_conditions_from_query(user_input, metadata_df, threshold=85):
    matched_conditions = []
    lowered_input = user_input.lower()

    for col in metadata_df.columns:
        for val in metadata_df[col].dropna().unique():
            if isinstance(val, str):
                score = fuzz.partial_ratio(val.lower(), lowered_input)
                if score >= threshold:
                    matched_conditions.append((col, val))
    return matched_conditions

def count_samples_by_group(project_path, condition): 
    metadata = load_metadata(f"{project_path}metadata.csv")
    counts = metadata[condition].value_counts().to_dict()
    return counts

ans= count_samples_by_group("/Users/camilla.callierotti/omics-agent/data/bulk_rna_seq/glio/", "Type")
print(ans)
# ### 2. COUNT MATRIX QUERIES ###

# def get_most_expressed_gene(project_path, condition):
#     counts = load_counts_matrix(project_path)
#     metadata = load_metadata(project_path)
    
#     samples = metadata[metadata['condition'] == condition]['sample_id']
#     condition_counts = counts[samples].mean(axis=1)
#     most_expr_gene = condition_counts.idxmax()
    
#     return f"The most expressed gene in {condition} is {most_expr_gene} with average count {condition_counts[most_expr_gene]:.2f}."


# def get_gene_expression(project_path, gene_name, condition):
#     counts = load_counts_matrix(project_path)
#     metadata = load_metadata(project_path)

#     samples = metadata[metadata['condition'] == condition]['sample_id']
    
#     if gene_name not in counts.index:
#         return f"Gene {gene_name} not found in the dataset."
    
#     expr_values = counts.loc[gene_name, samples]
#     return expr_values.to_dict()


# def get_deg_list(project_path, comparison_label="Gapmer2_vs_control"):
#     de_results = load_de_results(project_path)
    
#     if comparison_label not in de_results:
#         return f"No results found for comparison {comparison_label}."
    
#     df = de_results[comparison_label]
#     degs = df[df["padj"] < 0.05].sort_values("log2FoldChange", ascending=False)
#     return degs[["log2FoldChange", "padj"]].head(20).to_dict(orient="index")

# ### 3. ENRICHR RESULTS QUERIES ###

# def get_significant_genes(project_path, condition):
#     enrichr = load_enrichr_results(project_path)
    
#     if condition not in enrichr:
#         return f"No enrichR results found for condition {condition}."
    
#     sig_genes = enrichr[condition][enrichr[condition]['Adjusted P-value'] < 0.05]['Gene']
#     return sig_genes.tolist()

# ### 4. PLOT QUERIES ###

# def plot_pca(project_path):
#     pca_df = load_pca_data(project_path)
#     fig, ax = plt.subplots()
#     sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="condition", ax=ax)
#     ax.set_title("PCA of Samples")
#     return fig


# def plot_mds(project_path):
#     mds_df = load_mds_data(project_path)
#     fig, ax = plt.subplots()
#     sns.scatterplot(data=mds_df, x="Dim1", y="Dim2", hue="condition", ax=ax)
#     ax.set_title("MDS Plot of Samples")
#     return fig


# # if __name__ == "__main__":
# #     # Example test
# #     project_config = {
# #         "count_matrix": "data/fantom6/count_matrix.tsv",
# #         "metadata": "data/fantom6/metadata.csv"
# #     }

# #     # Try any of these:
# #     user_query = "Which genes are most expressed in Gapmer 2?"
# #     # user_query = "Top genes in untreated and treated"
# #     # user_query = "Most expressed in ctrl and gapmerr2"

# #     result = get_most_expressed_gene(user_query, project_config)
# #     print(result)
