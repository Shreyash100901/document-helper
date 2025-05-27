import os
from typing import List

from black import parse_ast
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader,DirectoryLoader
from firecrawl import FirecrawlApp
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from numpy.ma.core import true_divide

load_dotenv()
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
FIRECRAWL_API_KEY=os.environ.get('FIRECRAWL_API_KEY')

embeddings=OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


def ingest_docs():
    loader=ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")

    
    batch_size = 300  # You can adjust batch size as needed
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        PineconeVectorStore.from_documents(
            batch, embeddings, index_name="langchain-doc-index"
        )
        print(f"Uploaded batch {i // batch_size + 1} of {((len(documents) - 1) // batch_size) + 1}")
        
        
        
    PineconeVectorStore.from_documents(
        documents,embeddings, index_name="langchain-doc-index"
    )
    print("****Loading to vectorstore done ***")

# def ingest_docs2() -> None:
#     from langchain_community.document_loaders.firecrawl import FireCrawlLoader
#
#     langchain_documents_base_urls = [
#         "https://python.langchain.com/docs/integrations/chat/",
#         "https://python.langchain.com/docs/integrations/llms/",
#         "https://python.langchain.com/docs/integrations/text_embedding/",
#         "https://python.langchain.com/docs/integrations/document_loaders/",
#         "https://python.langchain.com/docs/integrations/document_transformers/",
#         "https://python.langchain.com/docs/integrations/vectorstores/",
#         "https://python.langchain.com/docs/integrations/retrievers/",
#         "https://python.langchain.com/docs/integrations/tools/",
#         "https://python.langchain.com/docs/integrations/stores/",
#         "https://python.langchain.com/docs/integrations/llm_caching/",
#         "https://python.langchain.com/docs/integrations/graphs/",
#         "https://python.langchain.com/docs/integrations/memory/",
#         "https://python.langchain.com/docs/integrations/callbacks/",
#         "https://python.langchain.com/docs/integrations/chat_loaders/",
#         "https://python.langchain.com/docs/concepts/",
#     ]
#
#     langchain_documents_base_urls2 = [
#         "https://python.langchain.com/docs/integrations/chat/"
#     ]
#     for url in langchain_documents_base_urls2:
#         print(f"FireCrawling {url=}")
#         loader = FireCrawlLoader(
#             url=url,
#             mode="crawl",
#             params={
#                 "crawlOptions":{"limit":5},
#                 "pageOptions":{"onlyMainContent":True},
#                 "waitUntilDone":True
#             }
#         )
#         docs = loader.load()
#
#         print(f"Going to add {len(docs)} documents to Pinecone")
#         PineconeVectorStore.from_documents(
#             docs, embeddings, index_name="firecrawl-index"
#         )
#         print(f"****Loading {url}* to vectorstore done ***")
#
# def ingest_docs3() -> List[Document]:
#     url = "https://python.langchain.com/docs/integrations/chat/"
#     app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
#
#     # Scrape the URL — formats can be markdown, html, text, etc.
#     result = app.scrape_url(url, formats=["markdown"])
#
#     # Extract the markdown text from the result
#     text = result.get("markdown", "")
#
#     # Wrap text in a LangChain Document with metadata
#     docs = [Document(page_content=text, metadata={"source": url})]
#
#     return docs


if __name__ == "__main__":
    documents = ingest_docs()