import os
import re
import yaml
import time
import random
from typing import Optional, Any, Dict
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


class RNASeqChatbotError(Exception):
    """Custom exception for chatbot errors"""
    pass


class APICapacityExceededError(RNASeqChatbotError):
    """Exception for API capacity/rate limit errors"""
    pass

class RNASeqChatbot:
    def __init__(self, model_name, api_key, db_uri, dialect, top_k, max_retries=3, base_delay=1):
        self.model_name = model_name
        self.api_key = api_key
        self.db_uri = db_uri
        self.dialect = dialect
        self.top_k=str(top_k)
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        self.llm = ChatMistralAI(api_key=api_key, model=model_name, temperature=0.2)
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
            input_variables=["question", "sql_query", "data", "columns", "plot_instructions", "plot_type", 'schema'],
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
    
    def _is_rate_limit_error(self, error_message: str) -> bool:
        """Check if the error is a rate limit or capacity exceeded error"""
        rate_limit_indicators = [
            "429",
            "rate limit",
            "capacity exceeded",
            "service tier capacity exceeded",
            "too many requests"
        ]
        error_lower = str(error_message).lower()
        return any(indicator in error_lower for indicator in rate_limit_indicators)

    def _exponential_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay (double waiting time after each retry)"""
        delay = self.base_delay * (2 ** attempt)
        # Add jitter to avoid thundering herd
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter

    def _invoke_chain_with_retry(self, chain: LLMChain, inputs: Dict[str, Any], operation_name: str) -> Any:
        """
        Invoke a LangChain with retry logic for rate limit errors
        
        Args:
            chain: The LangChain to invoke
            inputs: Input parameters for the chain
            operation_name: Name of the operation for logging
            
        Returns:
            Chain result
            
        Raises:
            APICapacityExceededError: If max retries exceeded for rate limit errors
            RNASeqChatbotError: For other persistent errors
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                print(f"Attempting {operation_name} (attempt {attempt + 1}/{self.max_retries + 1})")
                result = chain.invoke(inputs)
                if attempt > 0:
                    print(f"✅ {operation_name} succeeded after {attempt + 1} attempts")
                return result
                
            except Exception as e:
                last_error = e
                error_message = str(e)
                
                if self._is_rate_limit_error(error_message):
                    if attempt < self.max_retries:
                        delay = self._exponential_backoff_delay(attempt)
                        print(f"⚠️  Rate limit/capacity error in {operation_name}. Retrying in {delay:.2f} seconds...")
                        print(f"Error: {error_message}")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries exceeded for {operation_name} due to rate limiting")
                        raise APICapacityExceededError(
                            f"API capacity exceeded after {self.max_retries + 1} attempts for {operation_name}. "
                            f"Last error: {error_message}"
                        )
                else:
                    # Non-rate-limit error, don't retry
                    print(f"❌ Non-retryable error in {operation_name}: {error_message}")
                    raise RNASeqChatbotError(f"Error in {operation_name}: {error_message}")
        
        # This should never be reached, but just in case
        raise RNASeqChatbotError(f"Unexpected error in {operation_name}: {last_error}")
    
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
            return RNASeqChatbotError(f"SQL execution error: {str(e)}")
    
    def determine_plot_type(self, user_query, sql_query, data):
        """Determine what plot to generate, if any"""
        try:
            if data.empty:
                print("No data available for plotting.")
                return "no visualization"
        
            chain_result = self._invoke_chain_with_retry(
                self.plot_type_chain,
                {
                    "question": user_query,
                    "sql_query": sql_query,
                    "data": data.head().to_string(),
                    "columns": list(data.columns)
                },
            "plot type determination"
            )
            
            parsed_result = self._extract_structured_output(chain_result, self.plot_type_output_parser)
            plot_type = parsed_result["plot_type"]
                
            print(f"Determined plot type: {plot_type}")
            return plot_type
        
        except (APICapacityExceededError, RNASeqChatbotError):
            raise
        except Exception as e:
            print(f"Error determining plot type: {str(e)}")
            return "no visualization"
    
    def generate_plot(self, user_query, data, sql_query, plot_type):
        """
        Generate and execute plotly code
        """
        plotly_code = None
        try:
            if plot_type == "no visualization" or data.empty:
                print("No visualization requested or no data available.")
                return None
            
            # Get structured output
            data_as_dict = data.to_dict(orient='list')
            schema_info = {col: str(dtype) for col, dtype in data.dtypes.items()}
            plot_instructions = self.plot_instructions.get(plot_type, {})

            chain_result = self._invoke_chain_with_retry(
                self.plotly_chain,
                {
                    "question": user_query,
                    "data": data_as_dict,
                    "sql_query": sql_query,
                    "columns": list(data.columns),
                    "schema": schema_info,
                    "plot_type": plot_type,
                    "plot_instructions": plot_instructions
                },
                "plot code generation"
            )

            parsed_result = self._extract_structured_output(chain_result, self.python_code_output_parser)
            plotly_code = parsed_result["python_code"]
            
            print(f"Generated plotly code: {plotly_code}")
            
            exec_globals = {"px": px, "np": np, "data": data, "pd": pd}
            exec(plotly_code, exec_globals)
            
            if "fig" in exec_globals:
                return exec_globals["fig"]
                # Uncomment the following lines if you want to use via CLI
                # fig = exec_globals["fig"]
                # save = input("Save plot? (y/n): ")
                # if save.lower() == 'y':
                #     filename = input("Enter filename (without extension): ")
                #     fig.write_html(f"{filename}.html")
                #     print(f"Plot saved as {filename}.html")
                # return fig
            else:
                print("No figure object created in the plotly code.")
                return None
        
        except (APICapacityExceededError, RNASeqChatbotError):
            raise
        except Exception as e:
            print(f"Error generating/executing plot: {str(e)}")
            if plotly_code is not None:
                print(f"Problematic plotly code: {plotly_code}")
            else:
                print("Error occurred before plotly code generation.")
            return None

    def generate_response(self, user_query, data):
        """Generate natural language response"""
        try:
            if data.empty:
                return "No data was found matching your query. Please try rephrasing your question or check if the requested information exists in the database."
            
            # Prepare data summary for the LLM (limit size to avoid token limits)
            data_summary = data.head(10).to_string(index=False) if len(data) > 10 else data.to_string(index=False)
            data_info = f"Data shape: {data.shape}, Columns: {list(data.columns)}\n\nSample data:\n{data_summary}"
            
            chain_result = self._invoke_chain_with_retry(
                self.response_chain,
                {
                    "question": user_query,
                    "data": data_info
                },
                "response generation"
            )
            
            # response = self._extract_chain_output(chain_result, "response")
            
            parsed_result = self._extract_structured_output(chain_result, self.response_output_parser)
            response = parsed_result["analysis"]

            return response
            
        except (APICapacityExceededError, RNASeqChatbotError):
            raise
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response to your query."
    
    def chat(self, user_query):
        """Main chat function"""
        try:
            print(f"\n🧬 User query: {user_query}")
            
            # Step 1: Generate SQL query
            print("\n1. Generating SQL query...")
            sql_chain_result = self._invoke_chain_with_retry(
                self.sql_chain,
                {
                    "question": user_query,
                    "table_info": self.table_info,
                    "dialect": self.dialect,
                    "top_k": self.top_k
                },
                "SQL query generation"
            )
            
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
            #print(response)

            # Step 4: Check if visualization is needed
            print("\n4. Checking for visualization...")
            plot_type = self.determine_plot_type(user_query, sql_query, data)
            
            if plot_type != "no visualization":
                print(f"\n5. Generating {plot_type} plot...")
                self.generate_plot(user_query, data, sql_query, plot_type)
            
            return response
        
        except APICapacityExceededError as e:
            error_message = (
                "The Mistral API service capacity has been exceeded. "
                "This usually means the service is under heavy load. "
                "Please try again in a few minutes."
            )
            print(f"❌ API Capacity Error: {str(e)}")
            return error_message
            
        except RNASeqChatbotError as e:
            error_message = f"An error occurred while processing your query: {str(e)}"
            print(f"❌ Chatbot Error: {str(e)}")
            return error_message
            
        except Exception as e:
            error_message = "An unexpected error occurred while processing your query. Please try again or rephrase your question."
            print(f"❌ Unexpected Error: {str(e)}")
            return error_message
        
        
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