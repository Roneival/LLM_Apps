import os 
from minsearch import Index
from openai import OpenAI
from dotenv import load_dotenv
from gitsource import GithubRepositoryDataReader 


load_dotenv() 
openai_client = OpenAI(
    api_key=os.getenv("CEREBRAS_API_KEY"),
    base_url="https://api.cerebras.ai/v1"
)

def load_faq_data(): 

    documents = []

    reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path)

    files = reader.read()
    
    for file in files:
        doc = file.parse()
        documents.append(doc)

    return documents


def build_index(documents):
    index = Index(
        text_fields=["content"], 
        keyword_fields=["filename"]
        )
    index.fit(documents)
    return index