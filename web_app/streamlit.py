import streamlit as st
import os
from utils import load_data, read_article, invoking_agent, display_formatter, define_filters
from datetime import datetime

# Configuração inicial do layout em "wide mode"
st.set_page_config(page_title="Generator of summaries about Scientific Papers", layout="wide")

# Aplicação Streamlit
st.title("Tool to summarize Scientific Papers")
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose one option:", ["Visualize summarized papers", "Summarize New Paper"])

if option == "Visualize summarized papers":
    # Visualização de Artigos Sumarizados
    st.header("📚 Summarized Papers")
    st.subheader("Define the parameters to filter the papers.")
    start_date, end_date, flag_agenda, paper_theme = define_filters()

    st.write("Choose one paper by the title to see the summary and details.")

    # Carregar os dados
    df = load_data(start_date, end_date, flag_agenda, paper_theme)
    
    if df is not None:
        # Exibir todos os títulos no selectbox
        article_title = st.selectbox("Select one paper:", df["title"].tolist())

        # Filtrar o artigo selecionado
        selected_article = df[df["title"] == article_title].iloc[0]

        # Exibir detalhes do artigo selecionado
        display_formatter(selected_article)
    

elif option == "Summarize New Paper":
    # Sumarizar Novo Artigo
    st.header("📄 Summarize Article")
    st.write("Make upload of the PDF file, so the agent can generate the summary.")

    openai_key = st.sidebar.text_input("Add your OpenAI key:", type="password")
    langchain_key = st.sidebar.text_input("Add your LangChain key:", type="password")

    # Upload do arquivo PDF
    uploaded_file = st.file_uploader("Send your PDF file", type=["pdf"])
    
    if uploaded_file is not None and openai_key is not None and langchain_key is not None:
        # Configurar conexões
        try:
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["LANGCHAIN_API_KEY"] = langchain_key

        except Exception as e:
            st.error(f"Error to configurate the keys: {e}")

        # Ler o conteúdo do PDF
        try:
            from agent import compile_graph
            graph = compile_graph()
            
            with st.spinner("Processing the PDF file and generating article..."):
                article_info = read_article(uploaded_file)
                article_medium = invoking_agent(article_info, graph)

            # Exibir detalhes do artigo selecionado
            display_formatter(article_medium)
            
        except Exception as e:
            st.error(f"Error to process PDF file: {e}")