import os
import re
import yaml
from dotenv import load_dotenv
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import plotly.express as px # for llm-generated code execution
import numpy as np # for llm-generated code execution


class RNASeqChatbot:
    def __init__(self, model_name, api_key, db_uri, dialect, top_k):
        self.model_name = model_name
        self.api_key = api_key
        self.db_uri = db_uri
        self.dialect = dialect
        self.top_k=top_k
        
        self.llm = ChatMistralAI(api_key=api_key, model=model_name)
        self.db = SQLDatabase.from_uri(db_uri)
        self.table_info = self.db.get_table_info()
        
        # Load plot instructions from YAML file
        plot_instructions_path = os.getenv("CONFIG_YAML_PATH", "plotting_instructions.yaml")  
        if os.path.exists(plot_instructions_path):
            with open(plot_instructions_path, "r") as f:
                self.plot_instructions = yaml.safe_load(f)
        else:
            self.plot_instructions = {}

        # SQL generation prompt
        self.sql_prompt = PromptTemplate(
            input_variables=["question", "table_info", "dialect", "top_k"],
            template=
            """
            Given an input question on bulk RNA sequencing results from a research 
            scientist, create a syntactically correct {dialect} query to run to help 
            find the answer. Return only the SQL query, without any additional text 
            or explanation.
            
            Unless the user specifies in his question a specific 
            number of examples they wish to obtain, always limit your query to at 
            most {top_k} results. You can order the results by a relevant column to
            return the most interesting examples in the database. Always remove rows with
            null values.

            Never query for all the columns from a specific table, only ask for a the
            few relevant columns given the question. You may have to join tables to retrieve
            the relevant information. You can also use the `table_info` variable to see 
            the schema description of the database. If a column contains a dot (.) 
            in the name, wrap the column name in quotations (").
            
            If asked about counts or raw counts, use the table named normalization. If
            asked about columns which are not in the normalization table, try to find 
            them in the table named "metadata" and join it with the normalization table 
            to retrieve the relevant information.
            
            If asked about differential expression, and no sample subset is specified,
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

            Only use the following tables:
            {table_info}
            
            When generating queries, consider that the results might be used for 
            visualization. Include relevant columns that would be useful for plotting
            (like fold changes, p-values, gene names, sample information).
            
            User question: {question}
            """
        )
        
        # # Plot detection prompt
        # self.plot_detection_prompt = PromptTemplate(
        #     input_variables=["question", "data"],
        #     template=
        #     """
        #     You are an AI assistant to research scientists on bulk RNA-seq analyses.
        #     Based on the user question asked and the retreived data, would a plot 
        #     aid in understanding? Answer only "yes" or "no", nothing more.
            
        #     These sort of questions are likely to require a plot:
        #     1. Comparisons (e.g. "Is gene TGFB expressed more in treated cells than in 
        #     untreated?") -> Box Plot
        #     2. Aggregations and Summarizations (e.g., "What is the average expression per 
        #     condition?") → Box Plot
        #     3. Correlation (e.g. Is there correlation between log2FC and gene length?") 
        #     or PCA -> Scatter Plot
        #     4. Correlation between samples (e.g. Show me the correlation between samples.")
        #     -> Heatmap
        #     5. Differential expression (e.g. What are the 10 most significantly differentially 
        #     expressed genes in treated vs untreated samples?") -> Volcano Plot

        #     User question: {question}
        #     Retrieved data: {data}            
        #     """
        # )
        
        # Plotly code generation prompt
        self.plot_type_prompt = PromptTemplate(
            input_variables=["question", "sql_query", "data", "columns"],
            template=
            """
            You are an AI assistant to research scientists that recommends appropriate 
            data visualizations for bulk RNA-seq data analysis results. Based on the 
            user's question "{question}", SQL query {sql_query}, and query results {data}, 
            understand the most suitable type of graph or chart to visualize the data. 
            
            Available plot types and their use cases:
            1. Scatter: For visualizing relationships between two continuous variables,
            such as PCA scores or gene expression levels.
            2. Bar: For comparing counts or averages across categories
            3. Line: For showing trends over time or linear variables, such as
            expression changes across different time points or continuous conditions.
            4. Heatmap: For visualizing correlation matrices or expression patterns across
            multiple samples or genes.
            5. Box Plot: For visualizing the distribution of a continuous variable across
            different categories, such as expression levels across sample groups.
            6. Volcano Plot: For visualizing differential expression results, showing 
            fold change vs. significance.
            
            Respond only with the most appropriate plot type out of the available plot types
            above, and nothing else.
            Do not respond with any other type of plot or chart, and do not add anything
            else to the response. If the question does not match any of the above plot types,
            respond with "no visualization".
            """
        )
        
        
        # Plotly code generation prompt
        self.plotly_prompt = PromptTemplate(
            input_variables=["question", "sql_query", "data", "columns", "plot_instructions", "plot_type"],
            template=
            """
            You are an AI assistant to research scientists that produces appropriate 
            data visualizations for bulk RNA-seq data analysis results. Based on the 
            user's question "{question}", SQL query {sql_query}, and query results {data}, 
            produce a {plot_type} plot to visualize the data. 
            
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
        )
            
    
        # Response generation prompt
        self.response_prompt = PromptTemplate(
            input_variables=["question", "data"],
            template=
            """
            Generate a technical response aimed at a research scientist tailored to this 
            RNA-seq query based on the question asked: {question}, and data you retrieved: 
            {data}.
            The response should be concise, informative, and directly address the user's 
            query. Do not include any code or SQL queries or plots in the response, just 
            the analysis and findings.
            """
        )
        
        # Create chains
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        self.plot_type_chain = LLMChain(llm=self.llm, prompt=self.plot_type_prompt)
        self.plotly_chain = LLMChain(llm=self.llm, prompt=self.plotly_prompt)
        self.response_chain = LLMChain(llm=self.llm, prompt=self.response_prompt)
    
    def execute_sql(self, sql_query):
        """Execute SQL query and return results as DataFrame"""
        try:
            code = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql_query.strip(), flags=re.MULTILINE)
            df = pd.read_sql_query(code, self.db._engine)
            return df
        except Exception as e:
            return f"Error: {str(e)}"
    
    def plot_type(self, user_query, sql_query, data):
        """Determine what plot to generate, if any"""
        response = self.plot_type_chain.invoke({
            "question": user_query,
            "sql_query": sql_query,
            "data": data,
            "columns": data.columns
        })
        print(f"is_plot_query result: {response.get('text', '')}")  
        return response.get("text", "").lower() 
    
    def generate_plot(self, user_query, data, sql_query, plot_type, plot_instructions):
        """Generate and execute plotly code"""
        plotly_code = self.plotly_chain.invoke({
            "question": user_query,
            "data": data,
            "sql_query": sql_query,
            "columns": data.columns,
            "plot_type": plot_type,
            "plot_instructions": plot_instructions["plot_type"]
        })
        
        # Execute the generated plotly code
        exec_globals = {"px": px, "np": np, "data": data, "pd": pd}
        try:
            print(f"Generated plotly code before manipulation: {plotly_code.get('text', '')}")
            code = plotly_code.get("text", "")
            code_to_exec = re.sub(r"^```(?:python)?\s*|\s*```$", "", code.strip(), flags=re.MULTILINE)
            exec(code_to_exec, exec_globals)
        except Exception as e:
            print(f"Error executing plotly code: {e}")
            print("Generated code was: \n", code_to_exec)
            return
    
        # Get the figure from executed code
        if "fig" in exec_globals:
            fig = exec_globals["fig"]
            fig.show()
            
            # Option to save plot
            save = input("Save plot? (y/n): ")
            if save.lower() == 'y':
                filename = input("Enter filename (without extension): ")
                fig.write_html(f"{filename}.html")
                print(f"Plot saved as {filename}.html")
    
    def chat(self, user_query):
        """Main chat function"""
        # Convert user query to SQL
        sql_query = self.sql_chain.invoke({
            "question": user_query,
            "table_info": self.table_info,
            "dialect": self.dialect,
            "top_k": self.top_k
        })
        print(f"Generated SQL query: {sql_query.get('text')}")

        # Execute SQL query
        data = self.execute_sql(sql_query.get("text"))
        # if data.empty:
        #     print("Query returned no results.")
        #     return "No results found for your query."

        # Generate natural language response
        response = self.response_chain.invoke({
            "question": user_query,
            "data": data#.to_string(index=False)
        })
        print(f"Response: {response.get('text', '')}")
        
        # Check if plot is required
        plot_type = self.plot_type(user_query, sql_query.get("text"), data)
        if plot_type != "no visualization":
            self.generate_plot(user_query, data, sql_query, plot_type, self.plot_instructions)
        
        return response

