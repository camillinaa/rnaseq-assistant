import pandas as pd
import os
from glob import glob

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path, sep=",", index_col=0)

def load_counts(count_matrix_path):
    return pd.read_csv(count_matrix_path, sep="\t", index_col=0)

def load_deseq2_results(comparison_1, comparison_2):
    pattern = os.path.join(f"**/deseq2_toptable.Template_generic_{comparison_1}_vs_{comparison_2}.txt")
    #print(f"Searching for files with pattern: {pattern}")
    files = glob(pattern, recursive=True)
    #print(f"Found files: {files}")
    if not files:
        raise FileNotFoundError(f"No deseq2 result found for {comparison_1} vs {comparison_2}")
    return pd.read_csv(files[0], sep='\t')

def load_enrichr_results(comparison_1, comparison_2, direction='up'):
    if direction not in ['up', 'down', 'all']:
        raise ValueError("Direction must be 'up', 'down', or 'all'")
    pattern = os.path.join(f"**/enrichr.Template_generic_*{comparison_1}_vs_{comparison_2}_{direction}.xlsx")
    print(f"Searching for files with pattern: {pattern}")
    files = glob(pattern, recursive=True)
    print(f"Found files: {files}")
    if not files:
        raise FileNotFoundError(f"No Enrichr result found for {comparison_1} vs {comparison_2} ({direction})")
    return pd.read_excel(files[0], sheet_name=None)

#res = load_deseq2_results("Gapmer_1_Gapmer_2_Gapmer_3_Gapmer_4_Gapmer_5", "Ctrl")
# res = load_enrichr_results("Gapmer_1_Gapmer_2_Gapmer_3_Gapmer_4_Gapmer_5", "Ctrl", "down")
# res