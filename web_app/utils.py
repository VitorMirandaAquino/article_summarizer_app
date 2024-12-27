# Biblioteca para criação da aplicação
import streamlit as st

# Biblioteca para manipulação dos dados
import pandas as pd

# Biblioteca para extração do texto do PDF
from pdfminer.high_level import extract_text
# Biblioteca para uso de Templates para os prompts
import jinja2

# Biblioteca para configuração das estruturas do output das chamadas as LLMs
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from typing_extensions import Annotated, TypedDict


# Função para carregar os artigos já sumarizados
@st.cache_data
def load_data():
    df = pd.read_parquet('data/database_articles.parquet')
    return df
# Função para ler o pdf do artigo
@st.cache_data
def read_article(file_uploaded):
    article_text = extract_text(file_uploaded)
    return article_text

@st.cache_data
def messages_to_invoke_agent(article_text):

    # Prompt de sistema inicial
    template_path = "prompts/cleaner.jinja2"
    cleaner_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render()

    # Adicionando artigo ao dicionário de messagens
    initial_state = {
        "messages": [
            SystemMessage(content=cleaner_prompt, name="System"),
            HumanMessage(content=f"This is the article: \n {article_text}", name="User")
        ]
    }

    return initial_state

@st.cache_data
# Função para simular a sumarização do agente (substitua pela sua implementação)
def invoking_agent(initial_state, graph):
    # Aqui você pode integrar com o seu agente de sumarização
    output = graph.invoke(initial_state)

    return output

@st.cache_data
def display_formatter(graph_output):
    # Exibir detalhes do artigo selecionado
    tab1, tab2, tab3 = st.tabs(["Q&A", "Concepts", "Summary"])
    with tab1:
        st.title(graph_output['title'])
        st.markdown(graph_output['article_analysis'], unsafe_allow_html=True)
        
    with tab2:
        st.title(graph_output['title'])
        st.markdown(graph_output['concepts_medium'], unsafe_allow_html=True)

    with tab3:
        st.markdown(graph_output['summary_medium'], unsafe_allow_html=True)
        


