import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate

from datasets import load_dataset
import pandas as pd


from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.tools import Tool, BaseTool
from langchain_experimental.utilities import PythonREPL
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent,create_structured_chat_agent
from langchain.chains import LLMChain
from langchain.schema.runnable.config import RunnableConfig
import matplotlib.pyplot as plt
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional, Any
import re 
# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""
# ---- OPENAI LLM ---- #

llm = ChatOpenAI(model="gpt-4o")


# ---- Data ---- #
ds = load_dataset("aibabyshark/insurance_customer_support_QA_result")
df = pd.DataFrame(ds['train'])

# ---- TOOLS ---- #

#SQL tool


engine = create_engine("sqlite:///qa.db")
df.to_sql("qa_table",engine, if_exists='replace', index=False)
db = SQLDatabase.from_uri("sqlite:///qa.db")


sql_agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

sql_tool = Tool.from_function(
    name="sql_agent", 
    func=sql_agent.run, 
    description="useful for answering questions by using SQL"
) 

# python tool 
# Create directory if it doesn't exist
#if not os.path.exists('/mnt/data/'):
#    os.makedirs('/mnt/data/')


class PythonREPLTool(BaseTool):
    # ...

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)
        result = self.python_repl.run(query)

        # Check if the result is a matplotlib figure
        if isinstance(result, plt.Figure):
            # Save the figure to a file
            result.savefig('output.png')

        return result
    

python_tool = Tool.from_function(
    name="chart generator",
    func=PythonREPL().run,
    description="run python code to save charts for sql results in current working directory, do not open it",
    handle_tool_error = True,
    verbose = True
)

# feedback tool 

# Define your custom prompt template
prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template="{input_text}"
)

feedback_prompt = PromptTemplate(
    template="""Given the following question {input}.
    Provide training feedback for the customer support agent. 
    You must ask sql_agent to get the relevant qa_feedback_summary column from qa_table and then summarize it. 
    Write a detailed response based on the qa_feedback_summary column from qa_table only. 
    You should NOT generate your own data.
    You should NOT assume any data or returned data.
    You can end with a training plan for the customer support agent to improve the weakest area. 
    When confronted with choices, make a decision yourself with reasoning.
    """,
    input_variables=["input"],
)


llm_chain = LLMChain(llm=llm, prompt=feedback_prompt)


feedback_tool = Tool(
    name="feedback_agent",
    func=llm_chain.run,
    description="useful for providing training feedbacks based on qa_feedback_summary from sql_agent"
)

# -- AGENT -- #


tools = [sql_tool, python_tool, feedback_tool]
memory = ConversationBufferMemory(chat_history=ChatMessageHistory())

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION   ,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True
)

## -- GENERATION -- #
## Chainlit callback function
#@cl.set_starters
#async def starters():
#    return [
#        cl.Starter(
#            label="customer satisfaction analysis",
#            message="Generate a chart showing the month-to-month overall average satisfaction scores."
#        ),
#        cl.Starter(
#            label="problem solving analysis",
#            message="Who had an average score of problem solving lower than 5?"
#        ),
#        cl.Starter(
#            label="provide feedback",
#            message="Provide feedback to the agent with the lowest average score in March."
#        ),
#    ]
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("agent", agent)

@cl.on_message
async def process_user_query(message: cl.Message):
    agent = cl.user_session.get("agent")

    response = await agent.acall(message.content,
                                 callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(response["output"]).send()


    
    # Check if response["output"] contains ".png" and extract the filename
    output_text = response.get("output", "")
    
    # Use regex to find the pattern ending with .png
    match = re.search(r'\b\w+\.png\b', output_text)
    if match:
        filename = match.group(0)
        print("PNG Filename:", filename)
        file_path = os.path.join(os.getcwd(), filename)
        print(file_path)
        image = cl.Image(path=file_path, name="filename", display="inline", size = "large")
        await cl.Message(content="This message has an image:", elements=[image],).send()

    else:
        print("No PNG filename found in the text.")


if __name__ == "__main__":
    cl.run()

