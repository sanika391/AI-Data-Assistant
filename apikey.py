
# from langchain.agents import AgentType, initialize_agent
# from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
# from langchain.llms import GiminiPro
# from langchain_experimental.tools import PythonREPLTool


# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
# # from langchain_google.gemini import GeminiPro

# llm = GiminiPro()
# agent = create_python_agent(llm=llm, tool=PythonREPLTool())

# agent = initialize_agent(
#     tools=[agent],
#     llm=llm,
#     verbose=False,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
# )

# task = "print('Hello, World!')"

# result = agent.run(task)


# import pandas as pd
# df = pd.read_csv("data.csv")

# agent = create_pandas_dataframe_agent(llm, df)
# agent.run("How many rows are there?")