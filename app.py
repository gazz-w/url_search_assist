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
        temperature=1,
        openai_api_key=user_api_key,
    )

    prompt_template = """Using the company website at #company, create a concise and informative article for our sales team.
    The article should provide a comprehensive understanding of the prospect to aid in tailoring our sales approach. 
    Instructions:
    #Chain-of-Thought Analysis:

        -Company Overview:
        Examine the company's history, mission, and core values.
        Identify key leadership and decision-makers.

        -Products and Services:
        Summarize and organize in bullet points their main offerings.
        Highlight any unique selling propositions or innovative solutions.
        
        -Market Position:
        Determine their target markets and customer segments.
        Assess their competitive landscape and market share.
        
        -Recent News and Developments:
        Note any recent press releases, news articles, or significant announcements.
        Include partnerships, mergers, acquisitions, or expansions.
        
        -Potential Opportunities:
        Based on #my_company_area, and information gathered, suggest potential areas for collaboration or partnership.
        Identify challenges they face that our products/services can address.
        Suggest ways we could add value or differentiate from competitors.
        
    
    
    #Conclusion:
        - Summarize the key insights and recommend next steps for the sales team.

    #Specific Instructions:
        -Lenght: Keep the article between 400-600 words.
        -Tone: Professional and informative, suitable for a sales audience.
        -style: Use clear and concise language, avoiding jargon or technical terms. Organize content with headings and bullet points where appropriate.

    #Additional instructions:
        -Do not include any confidential or proprietary information not publicly available.
        -Write as if explaining to someone unfamiliar with the company.

        
    #Company: {context}
    #about_us_page: {about_context}
    #Output: {question} 
    #Must_contain: The company name; {must_contain}
    #My_company_area: {my_company_area}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question",
                         "must_contain", "my_company_area", "about_context"]
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    return llm_chain


def review_generated_text(generated_text, user_api_key):
    llm_reviewer = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=user_api_key,
    )

    review_prompt_template = """
    You are the reviewer of the AI-generated article.
    Please review the text below and ensure the content is structured as follows: [
        #Introduction about the Company Overview.
        #Main sections:
            Company Background.
            Products and services.
            Market Position.
            Recent News and Developments.
            Opportunities.
        #Conclusion
    ]

      
    #Specific Instructions:
    If the content is not structure correctly, make the necessary adjustments to ensure the article is informative and suitable for a sales audience.

    Organize content with headings. 
     
    Organize opportunities in bullet points where appropriate.

    Do not include or remove any information from the text, only restructure the content to meet the requirements.
    
    The outuput must be only the revised text.

    Texto para revisar:
    {generated_text}

    revised text:
    """

    prompt = PromptTemplate(
        template=review_prompt_template,
        input_variables=["generated_text"]
    )

    llm_chain = LLMChain(
        llm=llm_reviewer,
        prompt=prompt
    )

    response = llm_chain.run(generated_text=generated_text)

    return response


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
    my_company_area = st.sidebar.text_input(
        '3- Enter your company area and product')

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
                    about_context = about_docs = ""
                else:
                    st.write(f"About-us page: {about_url}")
                    st.success('creating the article...')
                    about_docs = get_documents_from_web(about_url)

                about_context = about_docs
                docs = get_documents_from_web(url)

                vector_store = create_vector_store(docs, user_api_key)
                chain = create_chain(user_api_key)

                retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                question = "produce a concise, informative article about this company for the sales team that {must_contain}"

                relevant_docs = retriever.get_relevant_documents(question)

                context = "\n\n".join(
                    [doc.page_content for doc in relevant_docs])

                response = chain.invoke({
                    "context": context,
                    "about_context": about_context,
                    "question": question,
                    "must_contain": must_contain,
                    "my_company_area": my_company_area,
                })

                article = response["text"]

                revised_article = review_generated_text(article, user_api_key)

                st.write(revised_article)
                st.download_button(label="download article", data=revised_article,
                                   file_name='article.txt', mime='text/plain')
                st.success('finished!')
            else:
                st.error(
                    "Please enter a valid URL starting with http:// or https://")
        except Exception as e:
            st.error(f"An error occurred:{e}")


if __name__ == "__main__":
    main()
