# Biblioteca para uso de Templates para os prompts
import jinja2

# Bibliotecas para obter variáveis de ambiente
import os, getpass

# Biblioteca para configuração das estruturas do output das chamadas as LLMs
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from typing_extensions import Annotated, TypedDict
from typing import Literal

# Biblioteca para conexão com a LLM
from langchain_openai import ChatOpenAI

# Biblioteca para conexão com banco NoSQL
from pymongo import MongoClient

# Biblioteca para definição do fluxo
from langgraph.graph import StateGraph, START, END

# Biblioteca para manipulação de datas
from datetime import datetime

# Biblioteca para aplicação web
import streamlit as st

# Conectar ao MongoDB
# Get the MongoDB URI from secrets
MONGO_URI = st.secrets["mongo"]["uri"]
client = MongoClient(MONGO_URI)
db = client["article_summarizer"]
collection = db["articles"]

# Configurando  Conexão com a OpenAI
llm_simples = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_complexa = ChatOpenAI(model="gpt-4o", temperature=0)

# States das mensagens
class OverallState(TypedDict):
    """All of the variables included in the flow"""
    first_page: Annotated[str, ...,"First page of the article for identification of the title"]
    article: Annotated[str, ..., "The full text of the article for graph flow"]
    title: Annotated[str, ..., "The title of the article"]
    cleaned_text: Annotated[str, ..., "The text without references, images and unnecessary information"]
    article_structure: Annotated[list, ...,"List of topics in the structure of the article"]
    concepts_explained: Annotated[list, ..., "The explanation of concepts involved in the article"]
    summary: Annotated[str, ..., "Summary of the article"]
    article_analysis: Annotated[str, ..., "Answers provided by the model over ther article"]
    concepts_medium: Annotated[str, ..., "Explanation of the key concepts formatted to be included in the streamlit"]
    summary_medium: Annotated[str, ..., "Summary of the article formatted to be included in the streamlit"]
    theme: Annotated[str, ..., "The theme of the article"]

class Raw_Article(TypedDict):
    """ First page of the article and all the content of the article"""
    first_page: Annotated[str, ...,"First page of the article for identification of the title"]
    article: Annotated[str, ..., "The full text of the article for graph flow"]

class Article_Title(TypedDict):
    """Title of the article"""
    title: Annotated[str, ..., "The title of the article"]

class Clean_Text(TypedDict):
    """Article text cleaned with the cleaned text and article structure."""
    cleaned_text: Annotated[str, ..., "The text without references, images and unnecessary information"]
    article_structure: Annotated[list, ...,"List of topics in the structure of the article"]

class Explain_Concepts(TypedDict):
    """Concepts explained"""
    concepts_explained: Annotated[list, ..., "The explanation of concepts involved in the article"]

class Article_Summary(TypedDict):
    """Summary of the article"""
    summary: Annotated[str, ..., "Summary of the article"]

class Article_Analysis(TypedDict):
    """Common questions aswered by the model"""
    article_analysis: Annotated[str, ..., "Answers provided by the model over ther article"]

class Article_Formatted(TypedDict):
    """Summary of the article"""
    concepts_medium: Annotated[str, ..., "Explanation of the key concepts formatted to be included in the streamlit"]
    summary_medium: Annotated[str, ..., "Summary of the article formatted to be included in the streamlit"]

class Article_Theme(TypedDict):
    """Theme of the article"""
    theme: Annotated[str, ..., "The theme of the article"]

def read_title(state: Raw_Article) -> Article_Title:
    initial_state = [
            SystemMessage(content="You are an assistant specialized in identify the title of scientific articles", name="System"),
            HumanMessage(content=f"This is the article: \n {state['first_page']}", name="User")
        ]

    structured_llm = llm_simples.with_structured_output(Article_Title)
    title_identified = structured_llm.invoke(initial_state)

    return title_identified

def verify_database(state: Article_Title) -> Literal["clean_article", "read_database"]:    
    # Verificar se o título do artigo já está no banco de dados
    documentos = list(collection.find({"title": state['title']}))

    # Verificar se a lista está vazia
    if not documentos:
        # Caso não tenha nenhum documento com o título, retornar "clean_article"
        return "clean_article"
    
    # Caso tenha algum documento com o título, retornar "read_database"
    return "read_database"

def read_database(state: OverallState) -> OverallState:
    documento = collection.find_one({"title": state['title']})

    state['cleaned_text'] = documento['cleaned_text']
    state['article_structure'] = documento['article_structure']
    state['concepts_explained'] = documento['concepts_explained']
    state["summary"] = documento["summary"]
    state["article_analysis"] = documento["article_analysis"]
    state["concepts_medium"] = documento["concepts_medium"]
    state["summary_medium"] = documento["summary_medium"]

    return state


# Função para processar o artigo e retornar um estado válido
def clean_article(state: Raw_Article) -> Clean_Text:
    ## Prompt de sistema inicial
    template_path = "prompts/cleaner.jinja2"
    cleaner_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render()
    
    # Adicionando artigo ao dicionário de messagens
    initial_state = [
            SystemMessage(content=cleaner_prompt, name="System"),
            HumanMessage(content=f"This is the article: \n {state['article']}", name="User")
        ]

    # Estrutura a saída com o formato Clean_Text
    structured_llm = llm_simples.with_structured_output(Clean_Text)
    cleaned_output = structured_llm.invoke(initial_state)

    # Retorna o estado atualizado no formato OverallState
    return cleaned_output

def explain_related_concepts(state: Clean_Text) -> Explain_Concepts:
    template_path = "prompts/concept_explainer.jinja2"
    concept_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render()
    initial_state = [
            SystemMessage(content=concept_prompt, name="System"),
            HumanMessage(content=f"This is the article: \n {state['cleaned_text']}", name="User")
        ]

    structured_llm = llm_complexa.with_structured_output(Explain_Concepts)
    concepts_explained = structured_llm.invoke(initial_state)

    return concepts_explained

def summarize_article(state: Clean_Text) -> Article_Summary:
    template_path = "prompts/summarizer.jinja2"
    summary_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render(article_structure="\n".join(state['article_structure']))
    initial_state = [
            SystemMessage(content=summary_prompt, name="System"),
            HumanMessage(content=f"This is the article: \n {state['cleaned_text']}", name="User")
        ]

    structured_llm = llm_complexa.with_structured_output(Article_Summary)
    summary_generated = structured_llm.invoke(initial_state)

    return summary_generated

def analyze_article(state: Clean_Text) -> Article_Analysis:
    template_path = "prompts/q&a.jinja2"
    q_a_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render()
    initial_state = [
            SystemMessage(content=q_a_prompt, name="System"),
            HumanMessage(content=f"This is the article: \n {state['cleaned_text']}", name="User")
        ]

    structured_llm = llm_complexa.with_structured_output(Article_Analysis)
    questions_answered = structured_llm.invoke(initial_state)

    return questions_answered
    
def format_medium_article(state: OverallState) -> OverallState:
    template_path = "prompts/formatter.jinja2"
    formatter_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render()
    provide_info = """
    This is the article:
    {0}
    ---
    This is the key concepts:
    {1}
    """.format(state['summary'], state['concepts_explained'])

    initial_state = [
            SystemMessage(content=formatter_prompt, name="System"),
            HumanMessage(content=provide_info, name="User")
        ]

    structured_llm = llm_simples.with_structured_output(Article_Formatted)
    medium_article = structured_llm.invoke(initial_state)

    return medium_article

# Função para processar o artigo e retornar um estado válido
def identify_theme(state: Explain_Concepts) -> Article_Theme:
    ## Prompt de sistema inicial
    template_path = "prompts/theme_identifier.jinja2"
    theme_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render()
    
    concepts_explained = '\n\n'.join(state['concepts_explained'])
    # Adicionando artigo ao dicionário de messagens
    initial_state = [
            SystemMessage(content=theme_prompt, name="System"),
            HumanMessage(content=f"This are the concepts contained in the article: \n {concepts_explained}", name="User")
        ]

    # Estrutura a saída com o formato Clean_Text
    structured_llm = llm_simples.with_structured_output(Article_Theme)
    theme_identified = structured_llm.invoke(initial_state)

    # Retorna o estado atualizado no formato OverallState
    return theme_identified

def save_database(state: OverallState) -> OverallState:
    execucao_fluxo = {
        "first_page": state["first_page"],
        "article": state["article"],
        "title": state["title"],
        "cleaned_text": state["cleaned_text"],
        "article_structure": state["article_structure"],
        "concepts_explained": state["concepts_explained"],
        "summary": state["summary"],
        "article_analysis": state["article_analysis"],
        "concepts_medium": state["concepts_medium"],
        "summary_medium": state["summary_medium"],
        "theme": state["theme"],
        "academia": False,
        "created_at": datetime.now()
    }
    collection.insert_one(execucao_fluxo)

    return state

def compile_graph():
    # Definindo grafo
    graph = StateGraph(OverallState, input=Raw_Article, output=OverallState)

    # Criando nós
    graph.add_node("read_title", read_title)
    graph.add_node("read_database", read_database)
    graph.add_node("clean_article", clean_article)
    graph.add_node("explain_related_concepts", explain_related_concepts)
    graph.add_node("summarize_article", summarize_article)
    graph.add_node("analyze_article", analyze_article)
    graph.add_node("format_medium_article", format_medium_article)
    graph.add_node("identify_theme", identify_theme)
    graph.add_node("save_database", save_database)

    # Definindo fluxo
    graph.add_edge(START, "read_title")
    graph.add_conditional_edges("read_title", verify_database)
    graph.add_edge("read_database", END)

    # Fluxo modelo simples
    graph.add_edge("clean_article", "explain_related_concepts")
    graph.add_edge("clean_article", "summarize_article")
    graph.add_edge("clean_article", "analyze_article")
    graph.add_edge("explain_related_concepts", "format_medium_article")
    graph.add_edge("summarize_article", "format_medium_article")
    graph.add_edge("analyze_article", "format_medium_article")

    # Fluxo modelo complexo
    #graph.add_edge("clean_article", "explain_related_concepts")
    #graph.add_edge("explain_related_concepts", "summarize_article")
    #graph.add_edge("summarize_article", "analyze_article")
    #graph.add_edge("analyze_article", "format_medium_article")


    graph.add_edge("format_medium_article", "identify_theme")
    graph.add_edge("identify_theme", "save_database")
    graph.add_edge("save_database", END)
    graph = graph.compile()

    return graph