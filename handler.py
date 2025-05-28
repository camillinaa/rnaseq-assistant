import pandas as pd
import pandasai as pai
from dotenv import load_dotenv
import os

# Set up the API key for pandasai
api_key = os.getenv("SECRET_API_KEY")
pai.api_key.set(api_key)

def run_counts_query(user_query, filepath):
    
    if not os.path.exists("datasets/rnaseq/counts/"):

        # Load data (this part can be substituted with connector to sql database)
        filepath = "data/normalization/cpm.txt"  # hardcoded path for counts matrix
        df = pd.read_csv(filepath, sep="\t")  # read with pandas for tab separator
        df.rename(columns={"Start": "start_pos", "End": "end_pos"}, inplace=True) # replace for sql conflict
        df["gene_name"] = df["gene_name"].astype("string")
        df["Geneid"] = df["Geneid"].astype("string")
        df["Chr"] = df["Chr"].astype("string")
        df["start_pos"] = df["start_pos"].astype("string")
        df["end_pos"] = df["end_pos"].astype("string")
        df["Strand"] = df["Strand"].astype("string")
        print(df.columns)
        
        pai_df = pai.DataFrame(df, name="counts_matrix")  # convert to pai dataframe

        # Create the semantic layer
        counts = pai.create(
            path="rnaseq/counts",
            df=pai_df,
            description="RNA-seq expression counts matrix",
            columns=[
                {"name": "gene_name", "type": "string", "description": "name of the gene"},
                {"name": "Geneid", "type": "string", "description": "Ensembl ID of the gene"},
                {"name": "Chr", "type": "string", "description": "chromosome"},
                {"name": "start_pos", "type": "string", "description": "start position"},
                {"name": "end_pos", "type": "string", "description": "end position"},
                {"name": "Strand", "type": "string", "description": "strand orientation"},
                {"name": "Length", "type": "integer", "description": "gene length"},
            ] + [
                {"name": col, "type": "float", "description": f"expression level for sample {col}"}
                for col in df.columns if col not in ["gene_name", "Geneid", "Chr", "start_pos", "end_pos", "Strand", "Length"]
            ]
        )
        
    df = pai.load("rnaseq/counts")

    response = df.chat(user_query)

    return response

def run_pca_mds_query(user_query, filepath):
    
    df = pd.read_csv(filepath, sep="\t")

    pai_df = pai.DataFrame(df, name="pca_scores")
    
    # if not os.path.exists("datasets/rnaseq/pca/"):
    #     # Create the semantic layer
    #     pca = pai.create(
    #         path="rnaseq/pca",
    #         df=pai_df,
    #         description="PCA scores for RNA-seq data",
    #         columns=[
    #             {"name": "samples", "type": "string", "description": "name of the sample"},
    #             {"name": "PC1", "type": "float", "description": "first principal component score"},
    #             {"name": "PC2", "type": "float", "description": "second principal component score"},
    #             {"name": "PC3", "type": "float", "description": "third principal component score"}
    #         ]
    #     )
        
    # df = pai.load("rnaseq/pca")
    
    response = pai_df.chat(user_query)
    
    return response

def run_deseq2_query(user_query, filepath):


    if not os.path.exists("datasets/rnaseq/deseq2/"):
        
        # Load data (this part can be substituted with connector to sql database)
        df = pd.read_csv(filepath, sep="\t") # read with pandas for tab separator
        df.rename(columns={"Start": "start_pos", "End": "end_pos"}, inplace=True) # replace for sql conflict
        print(df.columns)
        
        pai_df = pai.DataFrame(df, name="deseq2_table") # convert to pai dataframe

        # Create the semantic layer
        deseq2 = pai.create(
            path="rnaseq/deseq2",
            df=pai_df,
            description="gene expression data from DESeq2 analysis",
            columns=[
                {"name": "gene_name", "type": "string", "description": "name of the differentially expressed gene"},
                {"name": "baseMean", "type": "float", "description": "mean of normalized counts for all samples"},
                {"name": "log2FoldChange", "type": "float", "description": "log2 fold change between two conditions"},
                {"name": "pvalue", "type": "float", "description": "standard error of the log2 fold change"},
                {"name": "padj", "type": "float", "description": "adjusted p-value for multiple testing"},
                {"name": "Significance", "type": "string", "description": "direction of differential expression"},
                {"name": "Geneid", "type": "string", "description": "Ensembl gene ID"},
                {"name": "Chr", "type": "string", "description": "chromosome"},
                {"name": "start_pos", "type": "integer", "description": "start position"},
                {"name": "end_pos", "type": "integer", "description": "end position"},
                {"name": "Strand", "type": "string", "description": "strand orientation"},
                {"name": "Length", "type": "integer", "description": "gene length"}
            ]
        )

    df = pai.load("rnaseq/deseq2")

    response = df.chat(user_query)

    return response

# View code generated and executed locally by llm
# print(response.last_code_executed)