import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain

#  Retrieves documents from the web using the provided URL.


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500
    )
    split_docs = splitter.split_documents(docs)
    return split_docs

# create a vector database of documents


def create_vector_store(docs, user_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


# Create a chain that retrieves the answer to the user's question

def create_chain(user_api_key):
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=user_api_key,
    )

    prompt_template = """
    Your task is to research all the information about these company for a sales team.
    Company: {context}
    Output: {question} 
    Must contain: The company name; {must_contain}
    Must not contain: {must_not_contain}                                      
     """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question",
                         "must_contain", "must_not_contain"]
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    return llm_chain


def main():
    st.set_page_config(page_title="URL Search", page_icon="🔰")
    st.title("Sales Assistant 🔰")
    st.sidebar.header('LLM Configurations')
    st.sidebar.subheader(
        "Follow the instructions below to set up the Aplication.")

    user_api_key = st.sidebar.text_input(
        '1 - Enter your OpenAI API key', type='password')

    if not user_api_key:
        st.warning('Please set the API KEY')
        return

    st.sidebar.header('Search Preferences')
    must_contain = st.sidebar.text_input(
        '2- Enter the information that must be in the article')
    must_not_contain = st.sidebar.text_input(
        '3- Enter the information that you do not want in the article')

    url = st.text_input('Enter the URL')

    if not url:
        st.warning('Pleas enter the URL')
        return

    docs = get_documents_from_web(url)

    vector_store = create_vector_store(docs, user_api_key)
    chain = create_chain(user_api_key)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    question = "produce a concise, informative article about this prospect for the sales team"

    relevant_docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    response = chain.invoke({
        "context": context,
        "question": question,
        "must_contain": must_contain,
        "must_not_contain": must_not_contain,
    })

    article = response["text"]

    st.write(article)
    st.success('finished!')

    # to debug the response:
    # st.write(response)


if __name__ == "__main__":
    main()
