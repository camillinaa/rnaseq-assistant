import os
import re
import yaml
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# for llm-generated code execution:
import pandas as pd
import plotly.express as px 
import numpy as np 


class RNASeqChatbot:
    def __init__(self, model_name, db_uri, dialect, top_k, device=None, max_length=2048):
        self.db_uri = db_uri
        self.dialect = dialect
        self.top_k = top_k
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_length,
            device=0 if self.device == "cuda" else -1,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        
        self.db = SQLDatabase.from_uri(db_uri)
        self.table_info = self.db.get_table_info()
        
        # Load plot instructions from YAML file
        plot_instructions_path = os.getenv("CONFIG_YAML_PATH", "plotting_instructions.yaml")
        with open(plot_instructions_path, "r") as f:
            self.plot_instructions = yaml.safe_load(f)
            
        # Initialize output parsers
        self._setup_output_parsers()
        # Initialize prompts
        self._setup_prompts()

        def _setup_huggingface_model(self):
            """Initialize Hugging Face model and tokenizer"""
            try:
                print(f"Loading model: {self.model_name}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )
                
                # Add pad token if it doesn't exist
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                # Create text generation pipeline
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Create LangChain wrapper
                self.llm = HuggingFacePipeline(pipeline=self.pipe)
                
                print("Model loaded successfully!")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        
        self._setup_huggingface_model()

    def _setup_output_parsers(self):
            """Initialize structured output parsers for different chain outputs"""
            
            # SQL query parser
            sql_response_schemas = [
                ResponseSchema(name="sql_query", description="The SQL query to execute, without any markdown formatting or explanations")
            ]
            self.sql_output_parser = StructuredOutputParser.from_response_schemas(sql_response_schemas)
            self.format_sql_output_instructions = self.sql_output_parser.get_format_instructions()
            
            # Plot type parser
            plot_type_response_schemas = [
                ResponseSchema(name="plot_type", description="The recommended plot type: scatter, bar, line, heatmap, box, volcano, or no visualization")
            ]
            self.plot_type_output_parser = StructuredOutputParser.from_response_schemas(plot_type_response_schemas)
            self.format_plot_type_instructions = self.plot_type_output_parser.get_format_instructions()
            
            # Python code parser
            python_code_response_schemas = [
                ResponseSchema(name="python_code", description="The Python code to generate the plot, without any markdown formatting or explanations")
            ]
            self.python_code_output_parser = StructuredOutputParser.from_response_schemas(python_code_response_schemas)
            self.format_python_code_instructions = self.python_code_output_parser.get_format_instructions()
            
            # Response parser
            response_schemas = [
                ResponseSchema(name="analysis", description="Technical analysis and findings for the research scientist")
            ]
            self.response_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            self.format_response_instructions = self.response_output_parser.get_format_instructions()
            
    def _setup_prompts(self):
        """Initialize prompts for different tasks with output formatting"""

        # SQL generation prompt
        self.sql_prompt = PromptTemplate(
            input_variables=["question", "table_info", "dialect", "top_k"],
            partial_variables={"format_instructions": self.format_sql_output_instructions},
            template=
            """
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
        )
        
        # Visualization prompt
        self.plot_type_prompt = PromptTemplate(
            input_variables=["question", "sql_query", "data", "columns"],
            partial_variables={"format_instructions": self.format_plot_type_instructions},
            template=
            """
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
        )
        
        
        # Plotly code generation prompt
        self.plotly_prompt = PromptTemplate(
            input_variables=["question", "sql_query", "data", "columns", "plot_instructions", "plot_type"],
            partial_variables={"format_instructions": self.format_python_code_instructions},
            template=
            """
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
        )
            
    
        # Response generation prompt
        self.response_prompt = PromptTemplate(
            input_variables=["question", "data", "plot_type"],
            partial_variables={"format_instructions": self.format_response_instructions},
            template=
            """
            Generate a technical response aimed at a research scientist tailored to this 
            RNA-seq query based on the question asked: {question}, and data you retrieved: 
            {data}, and if applicable, the plot you generated.
            The response should be concise, informative, and directly address the user's 
            query. Do not include any code or SQL queries or plots in the response, just 
            the analysis and findings.
            Return a JSON object that adheres to the following schema: {format_instructions}
            """
        )
        
        # Create chains
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        self.plot_type_chain = LLMChain(llm=self.llm, prompt=self.plot_type_prompt)
        self.plotly_chain = LLMChain(llm=self.llm, prompt=self.plotly_prompt)
        self.response_chain = LLMChain(llm=self.llm, prompt=self.response_prompt)
        
        #self.parallel_chain = Chain([self.sql_chain, self.plot_type_chain, self.plotly_chain, self.response_chain])
    
    def _extract_structured_output(self, chain_result, parser):
        """Extract structured output using the appropriate parser"""
        if isinstance(chain_result, dict):
            raw_text = chain_result.get('text', str(chain_result))
        else:
            raw_text = str(chain_result)
        
        try:
            return parser.parse(raw_text)
        except Exception as e:
            print(f"Error parsing structured output: {str(e)}")
            print(f"Raw text: {raw_text}")
            # Fallback: try to extract JSON manually
            return self._fallback_json_extraction(raw_text, parser)
    
    def _fallback_json_extraction(self, raw_text, parser):
        """Fallback method to extract JSON from raw text"""
        import json
        
        # Try to find JSON in the text
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, raw_text)
        
        for match in matches:
            try:
                parsed_json = json.loads(match)
                # Check if it has the expected keys
                expected_keys = [schema.name for schema in parser.response_schemas]
                if any(key in parsed_json for key in expected_keys):
                    return parsed_json
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, return a default structure
        if hasattr(parser, 'response_schemas') and parser.response_schemas:
            key = parser.response_schemas[0].name
            return {key: raw_text.strip()}
        
        return {"error": "Could not parse response"}

    def execute_sql(self, sql_query):
        """Execute SQL query and return results as DataFrame"""
        try:
            #cleaned_query = self._clean_sql_query(sql_query)
            print(f"Executing cleaned SQL query: {sql_query}")
            df = pd.read_sql_query(sql_query, self.db._engine)
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
            
            parsed_result = self._extract_structured_output(chain_result, self.plot_type_output_parser)
            plot_type = parsed_result["plot_type"]
                
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
            if plot_type == "no visualization" or data.empty:
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
            
            # plotly_code_raw = self._extract_chain_output(chain_result, "plotly")
            # plotly_code = self._clean_python_code(plotly_code_raw)
            
            parsed_result = self._extract_structured_output(chain_result, self.python_code_output_parser)
            plotly_code = parsed_result["python_code"]
            
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
            print(f"Problematic plotly code: {plotly_code}")
            return None

    def generate_response(self, user_query, data):
        """Generate natural language response"""
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
            
            # response = self._extract_chain_output(chain_result, "response")
            
            parsed_result = self._extract_structured_output(chain_result, self.response_output_parser)
            response = parsed_result["analysis"]

            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response to your query."
    
    def chat(self, user_query):
        """Main chat function"""
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
            # parsed_sql_chain_result = self.parser.parse(sql_chain_result)
            # sql_query = parsed_sql_chain_result["sql"].strip()
            
            parsed_sql_result = self._extract_structured_output(sql_chain_result, self.sql_output_parser)
            sql_query = parsed_sql_result["sql_query"]
            
            if not sql_query:
                return "Failed to generate SQL query. Please try rephrasing your question."
            
            # Step 2: Execute SQL query
            print("\n2. Executing SQL query...")
            data = self.execute_sql(sql_query)
            if data.empty:
                return "No results found for your query. Please try rephrasing your question or check if the requested information exists in the database."
            print(f"Retrieved {len(data)} rows of data")

            # Step 3: Generate natural language response
            print("\n3. Generating response...")
            response = self.generate_response(user_query, data)
            print(response)

            # Step 4: Check if visualization is needed
            print("\n4. Checking for visualization...")
            plot_type = self.determine_plot_type(user_query, sql_query, data)
            
            if plot_type != "no visualization":
                print(f"\n5. Generating {plot_type} plot...")
                self.generate_plot(user_query, data, sql_query, plot_type)
            
            return 
        
        except Exception as e:
            print(f"Error in chat function: {str(e)}")
            return "An unexpected error occurred while processing your query. Please try again or rephrase your question."

if __name__ == "__main__":
    load_dotenv()

    # Load environment variables
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("MISTRAL_API_KEY")
    db_uri = os.getenv("DB_URI")
    dialect = os.getenv("DIALECT", "sqlite")
    top_k = int(os.getenv("TOP_K", 10))

    # Validate required environment variables
    if not api_key or not db_uri:
        raise ValueError("API_KEY and DB_URI environment variables must be set.")

    # Initialize chatbot
    chatbot = RNASeqChatbot(model_name, db_uri, dialect, top_k)

    print("RNASeqChatbot initialized. Type your query below:")
    while True:
        user_query = input("\n🧬 Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot.chat(user_query)
        print(f"\n🧬 Response: {response}")