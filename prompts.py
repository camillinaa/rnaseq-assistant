sql_prompt_template = """
    Given an input question on bulk RNA sequencing data, create a syntactically 
    correct {dialect} query to run to help find the answer. Return only the SQL 
    query, without any additional text or explanation. Return a JSON object that 
    adheres to the following schema: {format_instructions}.
    
    Unless the user specifies a specific number of examples they wish to obtain, 
    always limit your query to at most {top_k} results. You can order the results 
    by a relevant column to return the most interesting examples in the database. 
    Always remove rows with null values.

    Only use the following tables:
    {table_info}
    
    Never query for all the columns from a specific table, only ask for a the
    few relevant columns given the question. You may have to join tables to
    retrieve the relevant information. If a column contains a dot (.) in the 
    name, wrap the column name in quotations (").
    
    When asked about counts or raw counts, use the table named normalization. If
    asked about columns which are not in the normalization table, try to find 
    them in the table named "metadata" and join it with the normalization table 
    to retrieve the relevant information.
    
    When asked about differential expression, and no sample subset is specified,
    use the table containing "all_samples" in the name.
    
    When responding to questions about statistical significance, do not refer 
    to the raw pvalue column. Instead, use the adjusted p-value, which accounts 
    for multiple testing correction. The adjusted p-value column may be named 
    either padj or p.adjust, depending on the specific results table.
    
    If asked for information about specific samples, use the table metadata
    if necessary joining it with any other you deem necessary to retrieve
    the relevant information. In the metadata table, you can find the samples
    in the "Sample" column and the relative information to subset them in the
    other columns.

    Pay attention to use only the column names that you can see in the schema
    description. Be careful to not query for columns that do not exist. Also,
    pay attention to which column is in which table.
    
    When generating queries, consider that the results might be used for 
    visualization. Include relevant columns that would be useful for plotting
    (like fold changes, p-values, gene names, sample information).

    User question: {question}
    """

plot_type_prompt_template = """
    You are an AI assistant to research scientists that recommends appropriate 
    data visualizations for bulk RNA-seq data analysis results. Based on the 
    user's question "{question}", SQL query {sql_query}, and query results {data}, 
    understand the most suitable type of graph or chart to visualize the data. 
    Return a JSON object that adheres to the following schema: {format_instructions}
    
    Available plot types and their use cases:
    1. Scatter: For visualizing relationships between two continuous variables,
    such as PCA scores or gene expression levels.
    2. Bar: For comparing counts or averages across categories
    3. Line: For showing trends over time or linear variables, such as
    expression changes across different time points or continuous conditions.
    4. Heatmap: For visualizing correlation matrices or expression patterns across
    multiple samples or genes.
    5. Box: For visualizing the distribution of a continuous variable across
    different categories, such as expression levels across sample groups.
    6. Volcano: For visualizing differential expression results, showing 
    fold change vs. significance.
    
    Respond only with the most appropriate plot type out of the available plot types
    above, and nothing else.
    Do not respond with any other type of plot or chart, and do not add anything
    else to the response. If the question does not match any of the above plot types,
    respond with "no visualization".
    """

plotly_prompt_template = """
    You are an AI assistant to research scientists that produces appropriate 
    data visualizations for bulk RNA-seq data analysis results. Based on the 
    user's question "{question}", SQL query {sql_query}, and query results {data}, 
    produce a {plot_type} plot to visualize the data. Return a JSON object that 
    adheres to the following schema: {format_instructions}
    
    Use these detailed instructions for the selected plot type: {plot_instructions}.
    
    Give your response in the form of valid Python code that creates the appropriate 
    Plotly figure using plotly.express (aliased as px). The input data is provided as 
    a pandas DataFrame named data. Do not use df or any other variable name.
    You must only pass existing column names to the x and y arguments in the plotly 
    figure. Never pass an expression (e.g., np.log10(...)) as a string to x or y. If 
    a new column is needed, compute it first using NumPy (aliased as np), then pass 
    that new column name to the plot.
    All new columns must be added to data before the figure is created.
    Do not import any libraries. Do not include fig.show(). Use only px and np. Use 
    the exact column names provided in: {columns}. Title the plot meaningfully based 
    on the question: {question}. Only return code—do not include any explanations or 
    comments.
    """

response_prompt_template = """
    Generate a technical response aimed at a research scientist tailored to this 
    RNA-seq query based on the question asked: {question}, and data you retrieved: 
    {data}, and if applicable, the plot you generated.
    The response should be concise, informative, and directly address the user's 
    query. Do not include any code or SQL queries or plots in the response, just 
    the analysis and findings.
    Return a JSON object that adheres to the following schema: {format_instructions}
    """
