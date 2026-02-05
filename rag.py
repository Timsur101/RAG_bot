from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os

docs = []

folder = "data/docs"

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    
    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
        
    elif file.endswith(".txt"):
        loader = TextLoader(path)
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./db"
)

retriever = vectorstore.as_retriever()

llm = Ollama(model="llama3")

while True:
    query = input("\nВопрос: ")

    results = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
    Ответь на основе контекста:

    {context}

    Вопрос: {query}
    """

    response = llm.invoke(prompt)

    print("\nОтвет:")
    print(response)