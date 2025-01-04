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
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from typing_extensions import Annotated, TypedDict

# Biblioteca para conexão com banco NoSQL
from pymongo import MongoClient

# Função para carregar os artigos já sumarizados
#@st.cache_data
def load_data(start_date, end_date, flag_agenda, paper_theme):
    # Conexão com o banco de dados
    # Get the MongoDB URI from secrets
    MONGO_URI = st.secrets["mongo"]["uri"]
    client = MongoClient(MONGO_URI)
    db = client["article_summarizer"]
    collection = db["articles"]

    # Determinar os filtros
    filtros = []

    # Adicionar filtros dinamicamente
    if flag_agenda == "Yes":
        filtros.append({"academia": True})

    elif flag_agenda == "No":
        filtros.append({"academia": False})

    if paper_theme != "All":
        filtros.append({"theme": paper_theme})

    # Alguma condição
    filtros.append({
        "created_at": {
            "$gte": start_date,
            "$lt": end_date
        }
    })

    # Usar $and se houver múltiplos filtros
    consulta = {"$and": filtros} if filtros else {}

    if collection.count_documents(consulta) == 0:
        st.warning("No papers found with the selected filters.")
    else:
        documentos = collection.find(consulta)

        df = pd.DataFrame(documentos)
        return df
# Função para ler o pdf do artigo
@st.cache_data
def read_article(file_uploaded):
    # Extrai texto diretamente
    texto_extraido = extract_text(file_uploaded)
    primeira_pagina = extract_text(file_uploaded, maxpages=1)[:500]

    article_info = {
        "first_page": primeira_pagina,
        "article": texto_extraido
    }

    return article_info


# Função para simular a sumarização do agente (substitua pela sua implementação)
def invoking_agent(article_info, graph):
    # Aqui você pode integrar com o seu agente de sumarização
    try:
        output = graph.invoke(article_info)
    except Exception as e:
        st.error(f"Error invoking agent: {e}")

    return output


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
        


