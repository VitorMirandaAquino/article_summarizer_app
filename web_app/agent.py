# Biblioteca para uso de Templates para os prompts
import jinja2

# Biblioteca para configuração das estruturas do output das chamadas as LLMs
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from typing_extensions import Annotated, TypedDict

# Biblioteca para conexão com a LLM
from langchain_openai import ChatOpenAI

# Biblioteca para definição do fluxo
from langgraph.graph import StateGraph, START, END

# Configurando  Conexão com a OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# States das mensagens
class OverallState(TypedDict):
    """Todas as variáveis criadas pelo fluxo"""
    messages: Annotated[list[AnyMessage], add_messages]
    title: Annotated[str, ..., "The title of the article"]
    Cleaned_text: Annotated[str, ..., "The text without references, images and unnecessary information"]
    article_structure: Annotated[list, ...,"List of topics in the structure of the article"]
    concepts_explained: Annotated[list, ..., "The explanation of concepts involved in the article"]
    summary: Annotated[str, ..., "Summary of the article"]
    article_medium: Annotated[str, ..., "Article formatted to be included in the streamlit"]

class InputState(TypedDict):
    """ Messages sent to call the agent"""
    messages: Annotated[list[AnyMessage], add_messages]

class Clean_Text(TypedDict):
    """Article text cleaned with the title, cleaned text and article structure."""
    title: Annotated[str, ..., "The title of the article"]
    Cleaned_text: Annotated[str, ..., "The text without references, images and unnecessary information"]
    article_structure: Annotated[list, ...,"List of topics in the structure of the article"]

class Explain_Concepts(TypedDict):
    """Concepts explained"""
    concepts_explained: Annotated[list, ..., "The explanation of concepts involved in the article"]

class Article_Summary(TypedDict):
    """Summary of the article"""
    summary: Annotated[str, ..., "Summary of the article"]

class Article_Formatted(TypedDict):
    """Summary of the article"""
    article_medium: Annotated[str, ..., "Article formatted to be included in the streamlit"]


# Função para processar o artigo e retornar um estado válido
def clean_article(state: InputState) -> Clean_Text:
    # Estrutura a saída com o formato Clean_Text
    structured_llm = llm.with_structured_output(Clean_Text)
    cleaned_output = structured_llm.invoke(state['messages'])

    # Retorna o estado atualizado no formato OverallState
    return cleaned_output

def explain_related_concepts(state: Clean_Text) -> Explain_Concepts:
    template_path = "prompts/concept_explainer.jinja2"
    concept_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render()
    initial_state = [
            SystemMessage(content=concept_prompt, name="System"),
            HumanMessage(content=f"This is the article: \n {state['Cleaned_text']}", name="User")
        ]

    structured_llm = llm.with_structured_output(Explain_Concepts)
    concepts_explained = structured_llm.invoke(initial_state)

    return concepts_explained

def summarize_article(state: Clean_Text) -> Article_Summary:
    template_path = "prompts/summarizer.jinja2"
    summary_prompt = jinja2.Template(open(template_path, encoding="utf-8").read()).render(article_structure="\n".join(state['article_structure']))
    initial_state = [
            SystemMessage(content=summary_prompt, name="System"),
            HumanMessage(content=f"This is the article: \n {state['Cleaned_text']}", name="User")
        ]

    structured_llm = llm.with_structured_output(Article_Summary)
    summary_generated = structured_llm.invoke(initial_state)

    return summary_generated
    
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

    structured_llm = llm.with_structured_output(Article_Formatted)
    medium_article = structured_llm.invoke(initial_state)

    return medium_article


graph = StateGraph(OverallState, input=InputState, output=OverallState)
graph.add_node("clean_article", clean_article)
graph.add_node("explain_related_concepts", explain_related_concepts)
graph.add_node("summarize_article", summarize_article)
graph.add_node("format_medium_article", format_medium_article)
graph.add_edge(START, "clean_article")
graph.add_edge("clean_article", "explain_related_concepts")
graph.add_edge("clean_article", "summarize_article")
graph.add_edge("explain_related_concepts", "format_medium_article")
graph.add_edge("summarize_article", "format_medium_article")
graph.add_edge("format_medium_article", END)
graph = graph.compile()