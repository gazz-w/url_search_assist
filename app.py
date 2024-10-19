import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain.chains import create_retrieval_chain


def main_search():
    url = st.text_input('Enter the URL')

    if not url:
        st.warning('Please first set the API KEY and then enter the URL')
        return

    docs = get_documents_from_web(url)

    return docs

#############################################

#  Retrieves documents from the web using the provided URL.


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

# create a vector database of documents


def create_db(docs, user_api_key):
    embedding = OpenAIEmbeddings(api_key=user_api_key)
    vectorStore = faiss.FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

# Create a chain that retrieves the answer to the user's question


def create_chain(vectorStore, define_api_key):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=define_api_key,

    )

    prompt = ChatPromptTemplate.from_template("""
    Your task is to research all the information about these company for a sales team.
    Company: {context}
    Output: {input} 
    Must contain: {must_contain}
    Must not contain: {must_not_contain}                                      
    """)

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

#############################################


# docs = main_search()

# vectorStore = create_db(docs)

# chain = create_chain(vectorStore)


# response = chain.invoke({
#     "input": "quais são as últimas noticias",

# })

# st.write(response['answer'])


#############################################


def main():
    st.set_page_config(page_title="Url Search", page_icon="🔰")
    st.title("Sales Assistant 🔰")
    st.sidebar.header('LLM Configurations')
    st.sidebar.subheader(
        "Follow the instructions below to set up the Aplication model.")
    user_api_key = st.sidebar.text_input(
        '1 - Enter your OpenAI API key', type='password')

    st.sidebar.header('Search Preferences')
    must_contain = st.sidebar.text_input(
        '2- Enter the information that must be in the article')
    must_not_contain = st.sidebar.text_input(
        '3- Enter the information that you do not want in the article')

    docs = main_search()
    if docs:
        vectorStore = create_db(docs, user_api_key)
        chain = create_chain(vectorStore, user_api_key)

        response = chain.invoke({
            "input": "produce a concise, informative article about this prospect for the sales team",
            "must_contain": must_contain,
            "must_not_contain": must_not_contain,

        })

        st.write(response['answer'])
        st.success('finished!')


if __name__ == "__main__":
    main()
