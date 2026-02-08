from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama

import os


TOKEN = "tg_token"
DOCS_PATH = "./docs/KursachforRAG.pdf"
DB_PATH = "./chroma_db"

MAX_HISTORY = 6
users_memory = {}



def build_vector_db():

    print("Загрузка документации...")

    loader = DirectoryLoader(DOCS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )

    db.persist()

    print("Vector DB создана")



def load_db():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return db


def get_history(user_id):

    if user_id not in users_memory:
        users_memory[user_id] = []

    return users_memory[user_id]

def add_to_history(user_id, role, content):

    history = get_history(user_id)

    history.append({
        "role": role,
        "content": content
    })

    if len(history) > MAX_HISTORY:
        history.pop(0)


llm = Ollama(model="llama3")
db = None
retriever = None



def rag_chat(user_id, query):

    add_to_history(user_id, "user", query)

    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    history = get_history(user_id)

    history_text = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in history
    ])

    prompt = f"""
Ты помощник по документации продукции.
Будь вежливым и не груби, отвечай только в позитивном ключе.
Отвечай всегда на русском языке, никогда не переключайся на другие языки.
Отвечай по представленной документации.
Если ответа нет - скажи "Нет информации".
Не придумывай факты.
Отвечай чётко и структурировано.

История:
{history_text}

Контекст:
{context}

Вопрос:
{query}
"""

    response = llm.invoke(prompt)

    add_to_history(user_id, "assistant", response)

    return response



async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id
    text = update.message.text

    answer = rag_chat(user_id, text)

    await update.message.reply_text(answer)


def main():

    global db
    global retriever

    if not os.path.exists(DB_PATH):
        build_vector_db()

    db = load_db()

    retriever = db.as_retriever(
        search_kwargs={"k":4}
    )

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    print("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()