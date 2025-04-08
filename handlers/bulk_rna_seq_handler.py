import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re

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

def get_most_expressed_gene(user_input, project_config):
    # Load files
    count_matrix_path = project_config["count_matrix"]
    metadata_path = project_config["metadata"]

    counts_df = pd.read_csv(count_matrix_path, sep="\t")
    metadata_df = pd.read_csv(metadata_path)

    gene_id_col = "gene_name" if "gene_name" in counts_df.columns else counts_df.columns[0]
    sample_cols = counts_df.columns[2:]

    # Match conditions
    matched_conditions = find_conditions_from_query(user_input, metadata_df)

    if not matched_conditions:
        return "❌ No condition in your query matched the metadata."

    matching_samples = []
    mask = pd.Series(True, index=metadata_df.index)
    for col, val in matched_conditions: # filter metadata_df for samples that match the condition in query
        mask &= metadata_df[col] == val
    
    matched = metadata_df[mask]
    samples = matched["GF_ID"].tolist()
    matching_samples = [s for s in samples if s in sample_cols]

    if not matching_samples:
        return f"⚠️ No samples found for matched conditions: {matched_conditions}."

    # Subset and calculate mean expression
    counts_subset = counts_df[[gene_id_col] + matching_samples]
    counts_subset["mean_expr"] = counts_subset[matching_samples].mean(axis=1)

    # Sort by mean expression
    top_genes = counts_subset[[gene_id_col, "mean_expr"]].sort_values("mean_expr", ascending=False).head(10)

    # Return formatted result
    result = "\n".join(f"{row[gene_id_col]}: {row['mean_expr']:.2f}" for _, row in top_genes.iterrows())
    return f"🧬 Top expressed genes in {', '.join([v for _, v in matched_conditions])}:\n\n{result}"


if __name__ == "__main__":
    # Example test
    project_config = {
        "count_matrix": "data/fantom6/count_matrix.tsv",
        "metadata": "data/fantom6/metadata.csv"
    }

    # Try any of these:
    user_query = "Which genes are most expressed in Gapmer 2?"
    # user_query = "Top genes in untreated and treated"
    # user_query = "Most expressed in ctrl and gapmerr2"

    result = get_most_expressed_gene(user_query, project_config)
    print(result)
