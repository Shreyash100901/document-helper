from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "langchain-doc-index"

# Initialize Pinecone (make sure your API key and env are set)
pinecone_client=Pinecone(api_key=PINECONE_API_KEY, environment="us-east1-aws")

pinecone_client.delete_index(name=INDEX_NAME)
