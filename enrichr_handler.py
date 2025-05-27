import pandas as pd
import glob
import os
import spacy
from fuzzywuzzy import fuzz
import pandasai as pai
from pandasai import SmartDataframe

pai.api_key.set("PAI-95096dde-70c2-46e9-9aa0-2ff24003a3cc")

'''
assumes that the intent_classifier has already identified that the intent is enrichr
and the user has provided comparison_1 and comparison_2
'''

query = "Which pathways are significant?"

def get_enrichr_file(comparison_1, comparison_2, direction='up'):
    if direction not in ['up', 'down', 'all']:
        raise ValueError("Direction must be 'up', 'down', or 'all'")
    pattern = os.path.join(f"**/enrichr.Template_generic_*{comparison_1}_vs_{comparison_2}_{direction}.xlsx")
    print(f"Searching for files with pattern: {pattern}")
    files = glob.glob(pattern, recursive=True)
    print(f"Found files: {files}")
    if not files:
        raise FileNotFoundError(f"No Enrichr result found for {comparison_1} vs {comparison_2} ({direction})")
    return files[0]

# Load enrichr tables

sheet_names = ["GO_Biological_Process_2023", "GO_Cellular_Component_2023", "GO_Molecular_Function_2023", "Reactome_2022", "KEGG_2021_Human", "BioCarta_2016", "MSigDB_Hallmark_2020"]
file = get_enrichr_file("Gapmer_1_Gapmer_2_Gapmer_3_Gapmer_4_Gapmer_5", "Ctrl", "down")
enrichr = pd.DataFrame()
for sheet in sheet_names:
        df = pd.read_excel(file, sheet_name=sheet)
        df["ontology"] = sheet 
        enrichr = pd.concat([enrichr, df], ignore_index=True) 
enrichr.columns = [col.strip().lower().replace(".", "_") for col in enrichr.columns]

# Create the data layer

df = pai.DataFrame(enrichr)

enrichr = pai.create(
  path="rnaseq/enrichr",
  df=df,
  description="Dataset of pathway enrichment results",
      columns=[
        {
            "name": "term",
            "type": "string",
            "description": "name of the enriched gene set or pathway"
        },
        {
            "name": "overlap",
            "type": "string",
            "description": "proportion of input genes overlapping with the term"
        },
        {
            "name": "p_value",
            "type": "float",
            "description": "raw p-value of the enrichment test"
        },
        {
            "name": "adjusted_p_value",
            "type": "float",
            "description": "p-value adjusted for multiple testing"
        },
        {
            "name": "old_p_value",
            "type": "float",
            "description": "legacy version of the raw p-value"
        },
        {
            "name": "old_adjusted_p_value",
            "type": "float",
            "description": "legacy adjusted p-value"
        },
        {
            "name": "odds_ratio",
            "type": "string",
            "description": "how much more likely the term is enriched in your input set versus random chance"
        },
        {
            "name": "combined_score",
            "type": "string",
            "description": "composite score that combines the p-value and z-score, used by Enrichr to rank terms"
        },
        {
            "name": "genes",
            "type": "string",
            "description": "comma-separated list of input genes that hit the given term"
        },
        {
            "name": "ontology",
            "type": "string",
            "description": "source database or category of the term"
        }
    ]
)

# Configure the LLM
pai.config.set("temperature", 0)
pai.config.set("seed", 26)

query = "plot pvalue distribution"
# pandas-ai
pai.load("rnaseq/enrichr")
df.chat(query)
