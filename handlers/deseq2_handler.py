import pandas as pd
import pandasai as pai

def run_query(user_query, filepath):

    # Load data (this part can be substituted with connector to sql database)
    df = pd.read_csv(filepath, sep="\t") # read with pandas for tab separator
    df.rename(columns={"Start": "start_pos", "End": "end_pos"}, inplace=True) # replace for sql conflict
    print(df.columns)
    pai_df = pai.DataFrame(df, name="deseq2_table") # convert to pai dataframe

    # Create the semantic layer
    # deseq2 = pai.create(
    #     path="rnaseq/deseq2",
    #     df=pai_df,
    #     description="gene expression data from DESeq2 analysis",
    #     columns=[
    #         {"name": "gene_name", "type": "string", "description": "name of the differentially expressed gene"},
    #         {"name": "baseMean", "type": "float", "description": "mean of normalized counts for all samples"},
    #         {"name": "log2FoldChange", "type": "float", "description": "log2 fold change between two conditions"},
    #         {"name": "pvalue", "type": "float", "description": "standard error of the log2 fold change"},
    #         {"name": "padj", "type": "float", "description": "adjusted p-value for multiple testing"},
    #         {"name": "Significance", "type": "string", "description": "direction of differential expression"},
    #         {"name": "Geneid", "type": "string", "description": "Ensembl gene ID"},
    #         {"name": "Chr", "type": "string", "description": "chromosome"},
    #         {"name": "start_pos", "type": "integer", "description": "start position"},
    #         {"name": "end_pos", "type": "integer", "description": "end position"},
    #         {"name": "Strand", "type": "string", "description": "strand orientation"},
    #         {"name": "Length", "type": "integer", "description": "gene length"}
    #     ]
    # )

    # Load the saved data layer
    df = pai.load("rnaseq/deseq2")
    print(df.head())

    # Chat with dataset
    response = df.chat(user_query)

    return response

# View code generated and executed locally by llm
# print(response.last_code_executed)