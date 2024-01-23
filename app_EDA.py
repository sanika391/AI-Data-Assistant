import os
import re
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent, create_python_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.tools import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper


#-----------------------------------------------------------#

# Function to load and set API key
def load_api_key():
    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")
    
# Function to display welcome message and sidebar
def display_welcome():
    st.title("AI Assistant for Data Science 🤖")
    st.write('Hello,👋 I am your Assistant and I am here  to help you with your data science projects. 💚')
    
    # side bar 
    with st.sidebar:
        st.write('*Your Data Science Adventure Begins with an CSV File.* ')
        st.caption('''**You may already know that every exciting data science journey starts with
                a CSV file. Once we have your data in hand, 
                we'll dive into understanding it and have 
                some fun in exploring it. Then , we'll work 
                together to shape your business challenge
                into a data science framework. I'll introduce
                you to the coolest machine learning models,
                and we'll use them to tackle your problem.
                Sounds fun right**
                ''')
        # divider
        st.divider()
        st.caption('Made with love 💛', unsafe_allow_html=True)
 
# function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True 

# Function to handle user file upload
def handle_file_upload():
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv)
        return df
    return None

#-----------------------------------------------------------#

# Function to handle suggestions 
def suggestion_model(api_key, topic):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)    
    data_science_prompt = PromptTemplate.from_template("You are a genius data scientist. Write me a solution {topic}.")
    prompt_chain = LLMChain(llm=llm, prompt=data_science_prompt, verbose=True)
    resp = prompt_chain.run(topic)
    return resp

# # Function to load Wikipedia research based on the prompt
@st.cache_resource
def wiki(prompt):
    wiki_research = WikipediaAPIWrapper().run(prompt)
    return "Wikipedia Research for " + prompt

# Function to handle problem template chain
def prompt_templates():
    data_problem_template = PromptTemplate(
    input_variables=['business_problem'],
    template='Convert the following business problem into a data science problem: {business_problem}.'
    )
    template='''Give a list of machine learning algorithms with number sequence and as well as step by step 
    python code for any one algorithm that you think is suitable to solve 
    this problem: {data_problem}, while using this Wikipedia research: {wikipedia_research}.'''
    model_selection_template = PromptTemplate(
        input_variables=['data_problem', 'wikipedia_research'],
        template=template
    )

    return data_problem_template, model_selection_template

# Define the cache_data decorator for chains
@st.cache_data
def chains(_model):
    
    data_problem_chain = LLMChain(llm=_model, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
    model_selection_chain = LLMChain(llm=_model, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
    sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
    return sequential_chain

# Define the cache_data decorator for chains output
@st.cache_data
def chains_output(prompt, wiki_research, _model):
    my_chain = chains(_model)
    my_chain_output = my_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
    my_data_problem = my_chain_output["data_problem"]
    my_model_selection = my_chain_output["model_selection"]
    return my_data_problem, my_model_selection

# Function to extract machine learning algorithms from the output
@st.cache_data
def list_to_selectbox(input_text):
    algorithms_list = []
    lines = input_text.split('\n')

    for line in lines:
        # Use regular expression to find lines that seem to contain algorithm names
        match = re.search(r'\b([A-Za-z\s]+)\b', line)
        if match:
            algorithm_name = match.group(1).strip()
            algorithms_list.append(algorithm_name)

    # Insert "Select Algorithm" at the beginning
    # algorithms_list.insert(0, "Select Algorithm")

    return algorithms_list

# Function is part of the LangChain library and is used to create a Python Agent
# @st.cache_resource
# def python_agent(_model):
#     agent_executor = create_python_agent(
#         llm=_model,
#         tool=PythonREPLTool(),
#         verbose=True,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         handle_parsing_errors=True,
#         )
#     return agent_executor

# @st.cache_data
# def python_solution(my_data_problem, selected_algorithm, user_csv, _model):
#     solution = python_agent(_model).run(
#         f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}."
#     )
#     return solution
#----------------------------------------------------------#

# Function to diplay a overview of data
@st.cache_data(experimental_allow_widgets=True)
def data_overview(df, _pandas_agent):
    st.write("**Data Overview**")
    st.write("The first rows of your dataset look like this:")
    st.write(df.head())
    
    columns_df = _pandas_agent.run("What are the meaning of the columns?")
    if columns_df is not None:
        st.write(columns_df)
    else:
        st.warning("Unable to retrieve column information.")
    
    st.write("**Missing Values**")
    st.write("Number of missing values in each column:")
    st.write(df.isnull().sum())
    
    st.write("**Duplicate Values**")
    duplicates = _pandas_agent.run("Are there any duplicate values and if so where?")
    st.write(duplicates)
    
    st.write("**Data Summarisation**")
    st.write(df.describe())
    
    # Shape of the Dataset
    st.write("**Shape of the Dataset**")
    st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Skewness for Numeric Variables
    st.write("**Skewness for Numeric Variables**")
    numeric_columns = df.select_dtypes(include='number').columns
    for col in numeric_columns:
        skewness = df[col].skew()
        st.write(f"Skewness for '{col}': {skewness}")
    
    # # Data Types
    # st.write("**Data Types**")
    # selected_column_datatypes = st.selectbox("Select a column for data types:", df.columns)
    # if selected_column_datatypes:
    #     data_type = _pandas_agent.run(f"df['{selected_column_datatypes}'].dtypes")
    #     st.write(f"Data type of '{selected_column_datatypes}': {data_type}")
    # else:
    #     st.warning("Select a column to display its data type.")
    
    # # Visualizations
    # st.header("Visualizations")

    # # Histogram
    # st.subheader("Histogram")
    # selected_numeric_column = st.selectbox("Select a numeric column for histogram:", df.select_dtypes(include='number').columns)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.histplot(df[selected_numeric_column], ax=ax, color='#4CAF50')
    # st.pyplot(fig)

    # # Box Plot
    # st.subheader("Box Plot")
    # selected_numeric_column_box = st.selectbox("Select a numeric column for box plot:", df.select_dtypes(include='number').columns)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.boxplot(x=df[selected_numeric_column_box], color='#2196F3')
    # st.pyplot(fig)

    # # Pair Plot
    # st.subheader("Pair Plot")
    # selected_numeric_columns_pair = st.multiselect("Select numeric columns for pair plot:", df.select_dtypes(include='number').columns)
    # if selected_numeric_columns_pair:
    #     pair_plot = sns.pairplot(df[selected_numeric_columns_pair])
    #     st.pyplot(pair_plot.fig)
    # else:
    #     st.warning("Select at least two numeric columns for pair plot.")

    return  

# Function to perform user specific task
@st.cache_data(experimental_allow_widgets=True)
def perform_pandas_task(task, _pandas_agent):
    if task:
        return _pandas_agent.run(task)
    else:
        return f"Task '{task}' not recognized."

# Function to handle query
@st.cache_data(experimental_allow_widgets=True)
def perform_eda(input, _pandas_agent):
    if input:
        result = perform_pandas_task(input, _pandas_agent)
        st.write(f"**Result of '{input}'**")
        st.write(result)

# Function to display a brief information about a variable in a dataset. 
@st.cache_data
def variable_info(df, var):
    # Summary Statistics
    st.write(f"Summary Statistics for '{var}':")
    st.write(df[var].describe())
    
    # line plot
    st.line_chart(df, y=[var])

    # Distribution Visualization
    st.write(f"Distribution of '{var}':")
    fig, ax = plt.subplots()
    sns.histplot(df[var], kde=True, ax=ax, color='#4CAF50')
    st.pyplot(fig)

    # Box Plot
    st.write(f"Box Plot for '{var}':")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[var], ax=ax, color='#4CAF50')
    st.pyplot(fig)

    # Value Counts for Categorical Variables
    if df[var].dtype == 'O':  # Check if the variable is categorical
        st.write(f"Value Counts for '{var}':")
        st.write(df[var].value_counts())
    else:
        st.write(f"Outlier detection and normality tests are not applicable for variable  '{var}' .")

    # Outliers Detection
    st.write(f"Outliers Detection for '{var}':")
    if df[var].dtype != 'O':  # Check if the variable is not categorical
        z_scores = stats.zscore(df[var])
        outliers = df[(z_scores > 3) | (z_scores < -3)][var]
        st.write(outliers)
    else:
        st.write("Outlier detection is not applicable for categorical variables.")

    # Normality Test
    st.write(f"Normality Test for '{var}':")
    if df[var].dtype != 'O':  # Check if the variable is not categorical
        _, p_value = stats.normaltest(df[var].dropna())
        st.write(f"P-value: {p_value}")
        if p_value < 0.05:
            st.write("The variable does not follow a normal distribution.")
        else:
            st.write("The variable follows a normal distribution.")
    else:
        st.write("Normality test is not applicable for categorical variables.")

    # Missing Values
    st.write(f"Missing Values for '{var}':")
    st.write(df[var].isnull().sum())

    # Data Type
    st.write(f"Data Type for '{var}':")
    st.write(df[var].dtype)
    return 

# main
def main():
    GOOGLE_API_KEY=load_api_key()
    display_welcome()
    
    # initialise the key in session state
    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1:False}
    st.button("Let's get started", on_click=clicked, args=[1])
    if st.session_state.clicked[1]: 
        user_csv = handle_file_upload()
        if user_csv is not None:
            # Initialize pandas_agent
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
            pandas_agent = create_pandas_dataframe_agent(llm, user_csv, verbose=True)

            st.header("Exploratory Data Analysis")
            st.subheader("General information about dataset")
            data_overview(user_csv, pandas_agent)
                
        
    # with st.sidebar:
    #     with st.expander("What are the steps of EDA"):
    #         topic = 'What are the steps of Exploratory Data Analysis'
    #         resp = suggestion_model(GOOGLE_API_KEY, topic)
    #         st.write(resp)

    #     llm_suggestion = st.text_input("Ask me for a suggestion:")
    #     if llm_suggestion:
    #         llm_result = suggestion_model(GOOGLE_API_KEY, llm_suggestion)
    #         st.write(f"**LLM Suggestion:**")
    #         st.write(llm_result)
          
        
            st.subheader("Variable of study")
                
            user_question_variable = st.selectbox("What variable are you interested in?", user_csv.select_dtypes(include='number').columns)

            if user_question_variable:
                variable_info(user_csv, user_question_variable)
                st.subheader("Further Study")
                task_input = st.text_input("What task do you want to perform?")
                if task_input:
                    st.write(perform_pandas_task(task_input, pandas_agent))
    
                    st.divider()
                    st.header("Data Science Problem")
                    st.write("""Now that we have a solid grasp of the data at hand and a 
                            clear understanding of the variable we intend to investigate, 
                            it's important that we reframe our business 
                            problem into a data science problem.""")
                    
                    # Get user input
                    prompt = st.text_area('What is the business problem you would like to solve?')

                    # Display results on button click
                    if st.button("Get Suggestions"):
                        if prompt:
                            wiki_research = wiki(prompt)
                            my_data_problem, my_model_selection = chains_output(prompt, wiki_research, llm)
                            st.write("**Data Science Problem:**")
                            st.write(my_data_problem)
                            st.write("**Machine Learning Algorithm Suggestions:**")
                            st.write(my_model_selection)
                            algorithm_list = list_to_selectbox(my_model_selection)
                            # st.write(algorithm_list)
                            # selected_algorithm = st.selectbox("Select Machine Learning Algorithm", algorithm_list)
                            
                            # if selected_algorithm:
                            #     st.subheader("Assumption")
                            #     solution = python_solution(my_data_problem, selected_algorithm, user_csv, pandas_agent)
                            #     st.write(solution)
                                

















if __name__=='__main__':
    main()