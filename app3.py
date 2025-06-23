import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
tm = load_dotenv()

# Define prompt template components
custom_prefix = '''
You are a professional data analyst working with a pandas DataFrame that includes the following columns:
"Job_Title", "company", "Sector", "region", "technologies", "skills", "certifications", 
"soft_skills", "languages", "experience_years", "Type_Contrat", "level_study", "Tasks", "source"

Your task is to interpret the data in natural, user‐friendly language.
– Write your response in the user’s language (French if the user writes in French).
– Exclude any values marked "Not Specified" from your analysis.
– Do not include backend commentary or raw numbers; describe insights instead.
'''

format_instructions = '''
Use the following format:

Question: the input question you must answer
Thought: think through your approach
Action: one of [python_repl_ast]
Action Input: the python code to run
Observation: the output of that code
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the original question
'''

prompt_template = '''
You are working with a pandas dataframe named `df`.
Question: {{input}}
{{agent_scratchpad}}
'''

def init_agent(df: pd.DataFrame):
    """Initialize and return a pandas-DataFrame agent with custom prompts."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )

    agent = create_pandas_dataframe_agent(
        model,
        df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=[PythonAstREPLTool()],
        allow_dangerous_code=True,
        verbose=False,
        agent_executor_kwargs={"handle_parsing_errors": True},
        prefix=custom_prefix,
        format_instructions=format_instructions,
        suffix=prompt_template,
       # include_df_in_prompt=False,
    )
    return agent

# Cache data loading and agent creation
@st.cache_resource
def load_agent_and_data():
    df = pd.read_csv("DATA_PFE_VF_20_05_dash_chat.csv")
    agent = init_agent(df)
    return agent

agent = load_agent_and_data()

# Streamlit UI
st.set_page_config(page_title="Assistant Analyse Marché du Travail Marocain")
st.title("📊 Assistant Analyse Marché du Travail Marocain")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre assistant d'analyse du marché du travail marocain. Posez-moi vos questions sur les données."}
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question sur les données..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    placeholder = st.chat_message("assistant")
    with placeholder:
        thinking = st.empty()
        thinking.markdown("⌛ Réflexion en cours...")

    try:
        # Use the agent with the full prompt structure automatically
        response = agent.invoke(prompt)
        answer = response.get("output", "⚠️ Pas de réponse.")
    except Exception as e:
        answer = f"⚠️ Erreur lors du traitement : {e}"

    placeholder.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
