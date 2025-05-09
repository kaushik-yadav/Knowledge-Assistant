from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from config import OPENAI_API_BASE, OPENAI_API_KEY


def run_llm_agent(retriever, query):
    # initiating the LLM with model name and API credentials using an OPENAI wrapper
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    )
    # creating a system prompt for the agent
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say 'I don't know'. "
        "Use three sentences maximum. Be concise.\n\n"
        "Context:\n{context}"
    )

    # defining the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # creating chain of actions for documents
    # here we setup a question answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # combining the document retriever with the question answer chain that we created above
    # basically creating a RAG chain by combining the retriever and our question answer chain
    rag_chain: Runnable = create_retrieval_chain(retriever, question_answer_chain)

    # executing the RAG chain
    response = rag_chain.invoke({"input": query})
    return response["answer"]
