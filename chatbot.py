import os
import re
import yaml
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
from prompts import sql_prompt_template,plot_type_prompt_template, plotly_prompt_template, response_prompt_template
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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
        self.top_k=str(top_k)
        
        self.llm = ChatMistralAI(api_key=api_key, model=model_name)
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

        self.sql_prompt = PromptTemplate(
                input_variables=["question", "table_info", "dialect", "top_k"],
                partial_variables={"format_instructions": self.format_sql_output_instructions},
                template=sql_prompt_template
            )

        self.plot_type_prompt = PromptTemplate(
            input_variables=["question", "sql_query", "data", "columns"],
            partial_variables={"format_instructions": self.format_plot_type_instructions},
            template=plot_type_prompt_template
        )

        self.plotly_prompt = PromptTemplate(
            input_variables=["question", "sql_query", "data", "columns", "plot_instructions", "plot_type"],
            partial_variables={"format_instructions": self.format_python_code_instructions},
            template=plotly_prompt_template
        )

        self.response_prompt = PromptTemplate(
            input_variables=["question", "data", "plot_type"],
            partial_variables={"format_instructions": self.format_response_instructions},
            template=response_prompt_template
        )

        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        self.plot_type_chain = LLMChain(llm=self.llm, prompt=self.plot_type_prompt)
        self.plotly_chain = LLMChain(llm=self.llm, prompt=self.plotly_prompt)
        self.response_chain = LLMChain(llm=self.llm, prompt=self.response_prompt)
        
    def _extract_structured_output(self, chain_result, parser):
        """Extract structured output using the appropriate parser"""
        if isinstance(chain_result, dict):
            raw_text = chain_result.get('text', str(chain_result))
        else:
            raw_text = str(chain_result)
        
        return parser.parse(raw_text)
    
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
            
            return response
        
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
    chatbot = RNASeqChatbot(model_name, api_key, db_uri, dialect, top_k)

    print("RNASeqChatbot initialized. Type your query below:")
    while True:
        user_query = input("\n🧬 Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot.chat(user_query)
        print(f"\n🧬 Response: {response}")