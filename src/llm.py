import requests
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from config import OPENAI_API_BASE, OPENAI_API_KEY, OXFORD_APP_ID, OXFORD_APP_KEY
from src.rag import start_rag


# using the oxform API to get definitions of words, if not found the LLM will answer on its own.
def get_definition(word):
    url = f"https://od-api-sandbox.oxforddictionaries.com/api/v2/entries/en-us/{word.lower().strip()}"
    headers = {"app_id": OXFORD_APP_ID, "app_key": OXFORD_APP_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        try:
            definition = data["results"][0]["lexicalEntries"][0]["entries"][0][
                "senses"
            ][0]["definitions"][0]
            return definition
        except (KeyError, IndexError):
            return None
    return None


def run_llm_agent(retriever, query):
    # initiating the LLM with model name and API credentials using an OPENAI wrapper
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    )

    # if calculation is needed
    # ask the LLM if this is a calculation
    calc_check_prompt = (
        "Decide if this question requires performing a calculation. "
        "Reply only with YES or NO.\n\n"
        f"QUESTION: {query}"
    )
    should_calc = llm.invoke(calc_check_prompt).content.strip().upper()
    if should_calc == "YES":
        calc_prompt = (
            "Perform this calculation and return *only* the numeric result unless the user specifies to give better context:\n"
            f"{query}"
        )
        result = llm.invoke(calc_prompt).content.strip()
        return {
            "answer": result,
            "tool_used": "Tool: LLM Calculator",
            "context_snippets": [f"Calculated expression: {query} = {result}"],
        }

    # creating a system prompt for the agent
    system_prompt = (
        "You are a smart assistant with access to the following abilities:\n"
        "- If the question asks for a **definition**, you will be provided with a dictionary entry.\n"
        "- Use the definition (if given) to format the answer as per the user's query (e.g., return multiple meanings, add explanation).\n"
        "- If no definition is provided, use your own knowledge to answer.\n"
        "- For other queries, use the provided context.\n"
        "Be concise. Use at most 3 sentences.\n\n"
        "Context:\n{context}"
    )

    # defining the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # creating a chain of actions for documents
    # here we setup a question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # combining the document retriever with the question-answer chain that we created above
    # basically creating a RAG chain by combining the retriever and our question-answer chain
    rag_chain: Runnable = create_retrieval_chain(retriever, question_answer_chain)

    # instructing the LLM to extract the word to be defined dynamically if keywords like define are present
    if "define" in query.lower() or "definition" in query.lower():
        # ask the LLM to extract the word to be defined from the query
        extracted_word_response = llm.invoke(
            f"Extract the single word or phrase that needs to be defined from this question: '{query}'\n\nJust return the word or phrase without any explanation."
        )
        extracted_word = extracted_word_response.content.strip()
        # if the definition was extracted we output it else the LLM generates on its own
        if extracted_word:
            definition = get_definition(extracted_word)
            if definition:
                # adding a definition + llm layer to answer the questions
                definition_context = f"Definition of {extracted_word}: {definition}"
                final_answer = llm.invoke(
                    f"{definition_context}\n\nNow answer the user query: '{query}'"
                ).content.strip()

                return {
                    "answer": final_answer,
                    "tool_used": "Tool: Dictionary API + LLM Formatter",
                    "context_snippets": [definition_context],
                }
            else:
                # adding a fallback if we do not find the term in dictionary
                llm_definition = f"Define the term '{extracted_word}' and give two definitions if possible."
                answer = llm.invoke(llm_definition).content.strip()
                return {
                    "answer": f"This definition was not present in the dictionary, I will tell you what I know:\n{answer}",
                    "tool_used": "Tool: LLM generated definition",
                    "context_snippets": ["The definition was LLM generated"],
                }

    retrieved_docs = start_rag().get_relevant_documents(query)
    # executing the RAG chain
    response = rag_chain.invoke({"input": query})
    return {
        "answer": response["answer"],
        "tool_used": "Tool: RAG Pipeline",
        "context_snippets": [doc.page_content for doc in retrieved_docs],
    }
