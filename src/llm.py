import requests
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from config import OPENAI_API_BASE, OPENAI_API_KEY, OXFORD_APP_ID, OXFORD_APP_KEY


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

    # creating a system prompt for the agent
    system_prompt = (
        "You are a smart assistant with access to the following abilities:\n"
        "- If the question asks for a **definition**, try to extract the term from the query and fetch the definition from the Oxford API.\n"
        "- If the definition is not available through the API, generate the definition yourself based on your knowledge.\n"
        "- If the question asks you to **calculate**, perform the calculation yourself and return the answer.\n"
        "- If the question doesn't ask for a definition or calculation, use the provided context to answer the question.\n"
        "If you cannot answer the question with the context, say 'I don't know'.\n"
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

    # executing the RAG chain
    response = rag_chain.invoke({"input": query})

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
                return f"Definition of {extracted_word}: {definition}"
            else:
                print(
                    "This definition was not present in the dictionary, I will tell you what I know :"
                )

    return response["answer"]
