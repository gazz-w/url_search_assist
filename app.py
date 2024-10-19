import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import validators


# Exctract links with beautiful soup


def extract_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = []
    for link in soup.find_all('a', href=True):
        link_text = link.get_text(strip=True)
        href = link['href']
        full_url = urljoin(url, href)
        links.append({'text': link_text, 'url': full_url})
    return links

# Function to find the about us link


def find_about_us_link(user_api_key, url):
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=user_api_key,
    )

    # Extract links from the webpage
    links = extract_links(url)

    # Prepare the list of links as a string
    links_str = '\n'.join(
        [f"{i+1}. Text: {link['text']}, URL: {link['url']}" for i, link in enumerate(links)])

    # Define the prompt

    prompt_template = """
    You are to analyze the following list of webpage links and identify the URL of the 'About Us' page. The 'About Us' page may also be named 'About', 'Sobre Nós', 'Quem Somos', or similar variations in Portuguese. Provide only the URL of the correct page.
    Links: {links}
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["links", "format_instructions"]
    )

    class UrlExtract(BaseModel):
        url: str = Field(
            description="The exact URL of the 'About Us' or similar page")

    parser = JsonOutputParser(pydantic_object=UrlExtract)

    chain = prompt | llm | parser

    return chain.invoke({
        "links": links_str,
        "format_instructions": parser.get_format_instructions()
    })

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
    Must_contain: The company name; {must_contain}
    Article_structure example: {article_structure}                                      
     """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question",
                         "must_contain", "article_structure"]
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
    article_structure = st.sidebar.text_area(
        '3- Enter a model or structure that you want included in the article')

    url = st.text_input('Enter the URL')

    if not url:
        st.warning('Pleas enter the URL')
        return
    else:
        try:
            if validators.url(url):
                st.success("Valid URL")
                first_search = find_about_us_link(user_api_key, url)

                about_url = first_search["url"]
                if not about_url:
                    st.write(
                        f" not found, contiuing without the about us page...")
                    about_docs = ""
                else:
                    st.write(f"About-us page: {about_url}")
                    st.success('creating the article...')
                    about_docs = get_documents_from_web(about_url)

                docs = get_documents_from_web(url)

                vector_store = create_vector_store(docs, user_api_key)
                chain = create_chain(user_api_key)

                retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                question = "produce a concise, informative article about this company for the sales team that {must_contain}"

                relevant_docs = retriever.get_relevant_documents(question)

                context = "\n\n".join(
                    [doc.page_content for doc in relevant_docs])

                about_context = about_docs

                response = chain.invoke({
                    "context": context,
                    "about-page": about_context,
                    "question": question,
                    "must_contain": must_contain,
                    "article_structure": article_structure,
                })

                article = response["text"]

                st.write(article)
                st.success('finished!')
            else:
                st.error(
                    "Please enter a valid URL starting with http:// or https://")
        except Exception as e:
            st.error(f"An error occurred:")


if __name__ == "__main__":
    main()
