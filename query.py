import argparse
import time
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DB_DIR = "chroma_db"

def main():
    parser = argparse.ArgumentParser(description="Query the ingested PDFs.")
    parser.add_argument("query", type=str, help="The semantic query to execute.")
    args = parser.parse_args()

    print("Loading vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("Initializing LLM...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    system_prompt = (
        "You are an intelligent assistant for answering questions based on technical documentation. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer concise and relevant.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print(f"\nQuery: {args.query}\n")
    print("Searching and generating answer...")
    
    start_time = time.time()
    
    response = rag_chain.invoke({"input": args.query})
    
    end_time = time.time()
    retrieval_time = (end_time - start_time) * 1000

    print("\n--- Answer ---")
    print(response["answer"])
    print("\n--- Details ---")
    print(f"Time taken: {retrieval_time:.2f} ms")
    print("Sources used:")
    for doc in response["context"]:
        print(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})")

if __name__ == "__main__":
    main()
