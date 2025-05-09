from src.llm import run_llm_agent
from src.rag import start_rag

if __name__ == "__main__":
    query = input("Ask a question : ")
    retriever = start_rag()
    answer = run_llm_agent(retriever, query)
    print(answer)
