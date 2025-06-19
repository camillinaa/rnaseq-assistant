import pandas as pd
import pandasai as pai
from dotenv import load_dotenv
import os

# Set up the API key for pandasai
api_key = os.getenv("SECRET_API_KEY")
pai.api_key.set(api_key)


meta_df = pd.read_csv("data/metadata.csv", sep=",")
meta_df_wide = meta_df.set_index('Sample').T
counts_df = pd.read_csv("data/normalization/cpm.txt", sep="\t").pivot(index='Gene', columns='SampleID', values='Count')
pca_df = pd.read_csv("data/dim_reduction/PCA_scores.txt", sep="\t")
deseq_all_diff_df = pd.read_csv("data/dea_all_samples/dea_Type_E-GSC_L-GSC/deseq2_toptable.Type_E-GSC_vs_L-GSC.txt", sep="\t")
deseq_early_flat_df = pd.read_csv("data/dea_Early/dea_Flattening_Yes_No/deseq2_toptable.Flattening_Yes_vs_No.txt", sep="\t")

meta_pai = pai.DataFrame(meta_df, name="metadata")
counts_pai = pai.DataFrame(counts_df, name="count_matrix")
pca_pai = pai.DataFrame(pca_df, name="pca_scores")
deseq_all_diff_pai = pai.DataFrame(deseq_all_diff_df, name="differentially_expressed_genes_allsamples_e-gsc_l-gsc")
deseq_early_flat_pai = pai.DataFrame(deseq_early_flat_df, name="differentially_expressed_genes_earlysamples_flattening_yes_no")

response = pai.chat("what are the average counts of gene PALMD for the samples with flattening yes and no?", meta_pai, counts_pai, pca_pai) 
print(response.last_code_executed)
