import os
import streamlit as st
import pandas as pd

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

def load_analysis_table(selected_subset, selected_comparison, selected_analysis):
    """
    Load the table to query based on the analysis, subset, and comparison selected by the user on streamlit app.
    """
    subset = selected_subset.lower().replace(' ', '_')
    comparison = selected_comparison.lower().replace(' ', '_')
    root = "data"

    if selected_analysis == "deseq2":
        file_path = os.path.join(root, f"dea_{subset}", f"dea_{comparison}", f"deseq2_toptable.{comparison}.txt")
        with open(file_path, 'r') as file:
            df = pd.read_csv(file, sep="\t")
    elif selected_analysis == "ora":
        file_path = os.path.join(root, f"dea_{subset}", f"dea_{comparison}", f"ora_CP.{comparison}.all.txt")
        with open(file_path, 'r') as file:
            df = pd.read_csv(file, sep="\t")
    elif selected_analysis == "gsea":
        file_path = os.path.join(root, f"dea_{subset}", f"dea_{comparison}", f"gsea.{comparison}.xlsx")
        df = pd.read_csv(file_path)
    
    return df, file_path

# df = load_analysis_table("ns", "flattening yes no", "deseq2")
