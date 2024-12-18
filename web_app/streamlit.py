import streamlit as st
import os
from utils import load_data, read_article, messages_to_invoke_agent, invoking_agent


# Configuração inicial do layout em "wide mode"
st.set_page_config(page_title="Gerador de Resumos sobre Artigos Científicos", layout="wide")

# Aplicação Streamlit
st.title("Ferramenta de Sumarização de Artigos")
st.sidebar.title("Opções")
option = st.sidebar.selectbox("Escolha uma opção", ["Visualizar Artigos Sumarizados", "Sumarizar Novo Artigo"])

if option == "Visualizar Artigos Sumarizados":
    # Visualização de Artigos Sumarizados
    st.header("📚 Artigos Sumarizados")
    st.write("Escolha um artigo pelo título para visualizar o resumo e detalhes.")

    # Carregar os dados
    df = load_data()

    # Exibir todos os títulos no selectbox
    article_title = st.selectbox("Selecione um artigo:", df["title"].tolist())

    # Filtrar o artigo selecionado
    selected_article = df[df["title"] == article_title].iloc[0]

    # Exibir detalhes do artigo selecionado
    st.title(selected_article['title'])
    st.markdown(selected_article['article_medium'], unsafe_allow_html=True)  # Renderiza Markdown estilizado

elif option == "Sumarizar Novo Artigo":
    # Sumarizar Novo Artigo
    st.header("📄 Sumarizar Artigo")
    st.write("Faça upload de um arquivo PDF para que o agente possa gerar o resumo.")

    openai_key = st.sidebar.text_input("Adicione sua Key da OpenAI:", type="password")
    langchain_key = st.sidebar.text_input("Adicione sua Key da LangChain:", type="password")

    # Upload do arquivo PDF
    uploaded_file = st.file_uploader("Envie seu arquivo PDF", type=["pdf"])
    
    if uploaded_file is not None and openai_key is not None and langchain_key is not None:
        # Configurar conexões
        try:
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["LANGCHAIN_API_KEY"] = langchain_key
            

        except Exception as e:
            st.error(f"Erro ao configurar Keys: {e}")

        # Ler o conteúdo do PDF
        try:
            from agent import graph
            
            with st.spinner("Processando o arquivo PDF e gerando o artigo..."):
                article_text = read_article(uploaded_file)
                initial_state = messages_to_invoke_agent(article_text)
                article_medium = invoking_agent(initial_state, graph)

            # Exibir detalhes do artigo selecionado
            st.title(article_medium['title'])
            st.markdown(article_medium['article_medium'], unsafe_allow_html=True)  # Renderiza Markdown estilizado
        except Exception as e:
            st.error(f"Erro ao processar o arquivo PDF: {e}")