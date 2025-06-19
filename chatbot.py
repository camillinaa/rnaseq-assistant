from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from states import State, QueryOutput


class Chatbot:
    def __init__(self, db_uri: str, model_name: str, model_provider: str, api_key_env: str = "MISTRAL_API_KEY"):
        self.db = SQLDatabase.from_uri(db_uri)
        if not os.environ.get(api_key_env):
            os.environ[api_key_env] = getpass.getpass(f"Enter API key for {model_provider}: ")
        self.llm = init_chat_model(model_name, model_provider=model_provider)
        self.query_prompt_template = ChatPromptTemplate(
            [("system", self._system_message()), ("user", self._user_prompt())]
        )
        self.graph = self._build_graph()

    def _system_message(self):
        return """
        Given an input question on bulk RNA sequencing results from a research 
        scientist, create a syntactically correct {dialect} query to run to help 
        find the answer. Unless the user specifies in his question a specific 
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

        Pay attention to use only the column names that you can see in the schema
        description. Be careful to not query for columns that do not exist. Also,
        pay attention to which column is in which table.

        Only use the following tables:
        {table_info}
        """

    def _user_prompt(self):
        return "Question: {input}"

    def _build_graph(self):
        graph_builder = StateGraph(State).add_sequence(
            [self.write_query, self.execute_query, self.generate_answer]
        )
        graph_builder.add_edge(START, "write_query")
        return graph_builder.compile()

    def write_query(self, state: State):
        """Generate SQL query to fetch information."""
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": self.db.dialect,
                "top_k": 10,
                "table_info": self.db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = self.llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}

    def execute_query(self, state: State):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        return {"result": execute_query_tool.invoke(state["query"])}

    def generate_answer(self, state: State):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question, how you would answer "
            "it to a research scientist. Do not make any reference to the SQL queries.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content}

    def run_query(self, question: str):
        """
        Execute the complete query pipeline and return just the answer.
        
        Args:
            question: User's question about RNA-seq data
            
        Returns:
            String containing the final answer (not wrapped in dictionary)
        """
        try:
            # Execute the graph and collect all steps
            steps = list(self.graph.stream({"question": question}, stream_mode="updates"))
            # Find the step that contains the answer
            for step in reversed(steps):  # Start from the end to find the final answer
                if 'generate_answer' in step and 'answer' in step['generate_answer']:
                    return step['generate_answer']['answer']
            # Fallback if no answer found in expected format
            return "I encountered an issue processing your question. Please try rephrasing it."    
        except Exception as e:
            return f"An error occurred while processing your question: {e}"

