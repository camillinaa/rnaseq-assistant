import os
import re
import yaml
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain import Chain
# for llm-generated code execution:
import pandas as pd
import plotly.express as px 
import numpy as np 


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
        with open(plot_instructions_path, "r") as f:
            self.plot_instructions = yaml.safe_load(f)

        # SQL generation prompt
        self.sql_prompt = PromptTemplate(
            input_variables=["question", "table_info", "dialect", "top_k"],
            template=
            """
            Given an input question on bulk RNA sequencing results from a research 
            scientist, create a syntactically correct {dialect} query to run to help 
            find the answer. Return only the SQL query, without any additional text 
            or explanation.
            
            Unless the user specifies a specific number of examples they wish to obtain, 
            always limit your query to at most {top_k} results. You can order the results 
            by a relevant column to return the most interesting examples in the database. 
            Always remove rows with null values.

            Only use the following tables:
            {table_info}
            
            Never query for all the columns from a specific table, only ask for a the
            few relevant columns given the question. You may have to join tables to retrieve
            the relevant information. If a column contains a dot (.) 
            in the name, wrap the column name in quotations (").
            
            If asked about counts or raw counts, use the table named normalization. If
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
        )
        
        
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
        
        #self.parallel_chain = Chain([self.sql_chain, self.plot_type_chain, self.plotly_chain, self.response_chain])
    
    def _extract_chain_output(self, chain_result, chain_name=""):
        """
        Safely extract text content from LangChain output with proper error handling
        """
        try:
            if isinstance(chain_result, dict):
                return chain_result.get("text", "") or chain_result.get("content", "") or chain_result.get("output", "")
            elif isinstance(chain_result, str):
                return chain_result
            elif hasattr(chain_result, 'content'):
                content = chain_result.content
            elif hasattr(chain_result, 'text'):
                content = chain_result.text
            else:
                content = content.strip()
                if not content:
                    raise ValueError(f"Chain {chain_name} returned empty content")
            return content.strip()
        except Exception as e:
            raise ValueError(f"Error extracting output from chain {chain_name}: {str(e)}")
        
    def _clean_sql_query(self, sql_query):
        """
        Sanitize SQL query from LLM output
        """
        if not sql_query:
            raise ValueError("Empty SQL query recieved from LLM")
        code_block_pattern = r"```(?:sql)?\s*\n?(.*?)\n?```"
        match = re.search(code_block_pattern, sql_query, flags=re.MULTILINE | re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            keywords = ["SELECT", "WITH", "EXPLAIN"]
            lines = sql_query.strip().splitlines()
            sql_start_idx = 0
            sql_start_idx = len(lines)
            for i, line in enumerate(lines):
                if any(keyword in line.upper() for keyword in keywords):
                    sql_start_idx = i
                    break
            for i in range(len(lines) - 1, sql_start_idx - 1, -1):
                line = lines[i].strip()
                if line and (line.endswith(';') or 
                            any(keyword in line.upper() for keyword in ['FROM', 'WHERE', 'ORDER', 'GROUP', 'LIMIT', 'JOIN'])):
                    sql_end_idx = i + 1
                    break
            code = '\n'.join(lines[sql_start_idx:sql_end_idx]).strip()
        if not any(keyword in code.upper() for keyword in ["SELECT", "FROM"]):
            raise ValueError(f"Invalid SQL query generated: {code}")
        return code 
    
    def _clean_python_code(self, python_code):
        """
        Sanitize Python code from LLM output
        """
        if not python_code:
            raise ValueError("Empty Python code received from LLM")
        code_block_pattern = r"```(?:python)?\s*\n?(.*?)\n?```"
        match = re.search(code_block_pattern, python_code, flags=re.MULTILINE | re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            lines = python_code.strip().splitlines()
            # Find the first line that looks like Python code
            code_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if (stripped.startswith(('import ', 'from ', 'data[', 'fig =', 'px.')) or 
                    '=' in stripped or 
                    stripped.startswith('np.') or
                    stripped.startswith('pd.')):
                    code_start = i
                    break
            
            # Find the last line that looks like Python code
            code_end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                stripped = lines[i].strip()
                if (stripped and not stripped.startswith('#') and 
                    ('=' in stripped or stripped.startswith(('fig', 'data', 'px.', 'np.')))):
                    code_end = i + 1
                    break
            code = '\n'.join(lines[code_start:code_end]).strip()
        return code
        
    def _validate_plot_type(self, plot_type):
        """
        Validate and normalize plot type output
        """
        valid_types = ['scatter', 'bar', 'line', 'heatmap', 'box', 'volcano', 'no visualization']
        
        # Clean the plot type
        cleaned_type = plot_type.lower().strip()
        
        # Handle variations in naming
        type_mapping = {
            'scatterplot': 'scatter',
            'scatter plot': 'scatter',
            'barplot': 'bar',
            'bar chart': 'bar',
            'bar plot': 'bar',
            'lineplot': 'line',
            'line chart': 'line',
            'line plot': 'line',
            'heatmap plot': 'heatmap',
            'heat map': 'heatmap',
            'boxplot': 'box',
            'box plot': 'box',
            'volcano plot': 'volcano',
            'volcanoplot': 'volcano',
            'none': 'no visualization',
            'no plot': 'no visualization',
            'no chart': 'no visualization'
        }
        
        # Check direct match first
        if cleaned_type in valid_types:
            return cleaned_type
            
        # Check mapping
        if cleaned_type in type_mapping:
            return type_mapping[cleaned_type]
            
        # Check if any valid type is contained in the response
        for valid_type in valid_types:
            if valid_type in cleaned_type:
                return valid_type
                
        print(f"Warning: Unrecognized plot type '{plot_type}', defaulting to 'no visualization'")
        return 'no visualization'

    
    def execute_sql(self, sql_query):
        """
        Sanitize and execute SQL query and return results as DataFrame
        """
        try:
            cleaned_query = self._clean_sql_query(sql_query)
            print(f"Executing cleaned SQL query: {cleaned_query}")
            df = pd.read_sql_query(cleaned_query, self.db._engine)
            if df.empty:
                print("Query returned no results.")
                return pd.DataFrame()
            print(f"Query returned {len(df)} rows and {len(df.columns)} columns.")
            return df
        except Exception as e:
            print(f"Error executing SQL query: {str(e)}")
            print(f"Problematic query: {sql_query}")
            return pd.DataFrame()
    
    def determine_plot_type(self, user_query, sql_query, data):
        """Determine what plot to generate, if any"""
        try:
            if data.empty:
                print("No data available for plotting.")
                return "no visualization"
        
            chain_result = self.plot_type_chain.invoke({
                "question": user_query,
                "sql_query": sql_query,
                "data": data.head().to_string(),
                "columns": list(data.columns)
            })
            
            plot_type_raw = self._extract_chain_output(chain_result, "plot_type")
            plot_type = self._validate_plot_type(plot_type_raw)
                
            print(f"Determined plot type: {plot_type}")
            return plot_type
        
        except Exception as e:
            print(f"Error determining plot type: {str(e)}")
            return "no visualization"
    
    def generate_plot(self, user_query, data, sql_query, plot_type):
        """
        Generate and execute plotly code
        """
        try:
            if plot_type = "no visualization" or data.empty:
                print("No visualization requested or no data available.")
                return None
            
            plot_instructions = self.plot_instructions.get(plot_type, {})

            chain_result = self.plotly_chain.invoke({
                "question": user_query,
                "data": data.head().to_string(),
                "sql_query": sql_query,
                "columns": list(data.columns),
                "plot_type": plot_type,
                "plot_instructions": plot_instructions
            })
            
            plotly_code_raw = self._extract_chain_output(chain_result, "plotly")
            plotly_code = self._clean_python_code(plotly_code_raw)
            
            print(f"Generated plotly code: {plotly_code}")
            
            exec_globals = {"px": px, "np": np, "data": data, "pd": pd}
            
            exec(plotly_code, exec_globals)
            
            if "fig" in exec_globals:
                fig = exec_globals["fig"]
                fig.show()
                
                save = input("Save plot? (y/n): ")
                if save.lower() == 'y':
                    filename = input("Enter filename (without extension): ")
                    fig.write_html(f"{filename}.html")
                    print(f"Plot saved as {filename}.html")
                return fig
            else:
                print("No figure object created in the plotly code.")
                return None
        
        except Exception as e:
            print(f"Error generating/executing plot: {str(e)}")
            print(f"Problematic plotly code: {plotly_code_raw}")
            return None

    def generate_response(self, user_query, data):
        """
        Generate natural language response
        """
        try:
            if data.empty:
                return "No data was found matching your query. Please try rephrasing your question or check if the requested information exists in the database."
            
            # Prepare data summary for the LLM (limit size to avoid token limits)
            data_summary = data.head(10).to_string(index=False) if len(data) > 10 else data.to_string(index=False)
            data_info = f"Data shape: {data.shape}, Columns: {list(data.columns)}\n\nSample data:\n{data_summary}"
            
            chain_result = self.response_chain.invoke({
                "question": user_query,
                "data": data_info
            })
            
            response = self._extract_chain_output(chain_result, "response")
            
            if not response:
                return "I was able to retrieve data for your query, but encountered an issue generating the response. Please try rephrasing your question."
                
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response to your query."
    
    def chat(self, user_query):
        """
        Main chat function
        """
        try:
            print(f"\n🧬 User query: {user_query}")
            
            # Step 1: Generate SQL query
            print("\n1. Generating SQL query...")
            sql_chain_result = self.sql_chain.invoke({
                "question": user_query,
                "table_info": self.table_info,
                "dialect": self.dialect,
                "top_k": self.top_k
            })
            sql_query = self._extract_chain_output(sql_chain_result, "SQL generation")
            if not sql_query:
                return "Failed to generate SQL query. Please try rephrasing your question."
            
            # Step 2: Execute SQL query
            print("\n2. Executing SQL query...")
            data = self.execute_sql(sql_query.)
            if data.empty:
                return "No results found for your query. Please try rephrasing your question or check if the requested information exists in the database."
            print(f"Retrieved {len(data)} rows of data")

            # Step 3: Generate natural language response
            print("\n3. Generating response...")
            response = self.generate_response(user_query, data)

            # Step 4: Check if visualization is needed
            print("\n4. Checking for visualization...")
            plot_type = self.determine_plot_type(user_query, sql_query, data)
            
            if plot_type != "no visualization":
                print(f"\n5. Generating {plot_type} plot...")
                self.generate_plot(user_query, data, sql_query, plot_type)
            
            return response
        
        except Exception as e:
            print(f"Error in chat function: {str(e)}")
            return "An unexpected error occurred while processing your query. Please try again or rephrase your question."
