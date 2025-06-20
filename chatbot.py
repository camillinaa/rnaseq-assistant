import os
import re
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


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
            return the most interesting examples in the database.

            Never query for all the columns from a specific table, only ask for a the
            few relevant columns given the question. You may have to join tables to retrieve
            the relevant information. You can also use the `table_info` variable to see 
            the schema description of the database. If a column contains a dot (.) 
            in the name (e.g. "p.adjust"), wrap the column name in quotations (").
            
            If asked about counts or raw counts, use the table named normalization. If
            asked about columns which are not in the normalization table, try to find 
            them in the table named "metadata" and join it with the normalization table 
            to retrieve the relevant information.
            
            If asked about differential expression, and no sample subset is specified,
            use the table containing "all_samples" in the name.
            
            If asked about significance, don't use the pvalue column, but rather the 
            p.adjust column.
            
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
        
        # Plot detection prompt
        self.plot_detection_prompt = PromptTemplate(
            input_variables=["question", "data"],
            template=
            """
            You are an AI assistant to research scientists on bulk RNA-seq analyses.
            Based on the user question asked and the retreived data, would a plot 
            aid in understanding? Answer only "yes" or "no", nothing more.
            
            These sort of questions are likely to require a plot:
            1. Comparisons (e.g. "Is gene TGFB expressed more in treated cells than in 
            untreated?") -> Box Plot
            2. Aggregations and Summarizations (e.g., "What is the average expression per 
            condition?") → Box Plot
            3. Correlation (e.g. Is there correlation between log2FC and gene length?") 
            or PCA -> Scatter Plot
            4. Correlation between samples (e.g. Show me the correlation between samples.")
            -> Heatmap
            5. Differential expression (e.g. What are the 10 most significantly differentially 
            expressed genes in treated vs untreated samples?") -> Volcano Plot

            User question: {question}
            Retrieved data: {data}            
            """
        )
        
        # Plotly code generation prompt
        self.plotly_prompt = PromptTemplate(
            input_variables=["question", "data"],
            template=
            """
            You are an AI assistant to research scientists that recommends appropriate 
            data visualizations for bulk RNA-seq data analysis results. Based on the 
            user's question, SQL query, and query results, understand the most suitable 
            type of graph or chart to visualize the data. 
            
            Give your response in the format of python code to create the appropriate 
            plotly figure. Provide only the code and nothing more. In the Python code, 
            assume the SQL query result is stored in a DataFrame called data, not df.
            Do not include import plotly.express as px nor fig.show() in the code. Give
            the plot a meaningful title based on the question asked.
            If no visualization is appropriate, respond with "no visualization".

            Available chart types and their use cases:
            1. Volcano Plot: For visualizing differential expression results, showing 
            fold change vs. significance.
            2. Scatter Plot: For visualizing relationships between two continuous variables,
            such as PCA scores or gene expression levels.
            3. Heatmap: For visualizing correlation matrices or expression patterns across
            multiple samples or genes.
            4. Box Plot: For visualizing the distribution of a continuous variable across
            different categories, such as expression levels across sample groups.
            5. Line Chart: For showing trends over time or ordered categories, such as
            expression changes across different time points or conditions.
            
            User question: {question}
            Retrieved data: {data}
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
        self.plot_detection_chain = LLMChain(llm=self.llm, prompt=self.plot_detection_prompt)
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
    
    def is_plot_query(self, user_query, data):
        """Determine if query is asking for a plot"""
        response = self.plot_detection_chain.invoke({
            "question": user_query,
            "data": data
        })
        print(f"is_plot_query result: {response.get('text', '')}")  
        return "yes" in response.get("text", "").lower() # outputs a boolean
    
    def generate_plot(self, user_query, data):
        """Generate and execute plotly code"""
        plotly_code = self.plotly_chain.invoke({
            "question": user_query,
            "data": data
        })
        
        # Execute the generated plotly code
        exec_globals = {"px": px, "data": data, "pd": pd}
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
        
        # Check if plot is requested
        if self.is_plot_query(user_query, data):
            self.generate_plot(user_query, data)
        
        return response

# Usage example
if __name__ == "__main__":
    # Initialize environment variables and database
    load_dotenv() 
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("Please set your Mistral API key in the .env file.")
    
    # Initialize chatbot
    chatbot = RNASeqChatbot(
        api_key=os.getenv("MISTRAL_API_KEY"),
        db_uri="sqlite:///rnaseq.db"
    )
    
    # Chat loop
    while True:
        user_input = input("Ask about RNA-seq data (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        try:
            chatbot.chat(user_input)
        except Exception as e:
            print(f"Error: {e}")