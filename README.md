# URL Search

An AI-powered workflow that automates prospect company research for a sales team. The workflow takes a company website as input and produces an informative article about this prospect as output.

## Features

- Extracts every link using Beautiful Soup to find the "about-us" URL.
- Extracts information from both the provided page and the "about-us" page.
- Passes this information to an LLM model to generate a detailed article about the company.
- Download the article as a txt file.
- Supports multiple languages.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To run the application locally

```bash
streamlit run app.py
```

And follow the instructions in the web app.

You will need an OpenAI API key.

Another way to access the app is through the Streamlit direct link: [https://urlsearchassist.streamlit.app/](https://urlsearchassist.streamlit.app/)

## Demo Video

[Assista à demonstração do projeto](https://www.linkedin.com/feed/update/urn:li:activity:7303430159213170688/)




