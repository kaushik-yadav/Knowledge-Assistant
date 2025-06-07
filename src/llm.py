import os
from functools import lru_cache

import requests
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# load environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_base = os.environ.get("OPENAI_API_BASE")
oxford_app_id = os.environ.get("OXFORD_APP_ID")
oxford_app_key = os.environ.get("OXFORD_APP_KEY")

# initiating the LLM with model name and API credentials using an OPENAI wrapper
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
)

# using the oxford API to get definitions of words, if not found the LLM will answer on its own.
@lru_cache(maxsize=128)
def get_definition(word):
    url = f"https://od-api-sandbox.oxforddictionaries.com/api/v2/entries/en-us/{word.lower().strip()}"
    headers = {"app_id": oxford_app_id, "app_key": oxford_app_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        try:
            definition = data["results"][0]["lexicalEntries"][0]["entries"][0]["senses"][0]["definitions"][0]
            return definition
        except (KeyError, IndexError):
            return None
    return None

# Create a dictionary to store chat histories per user/session
chat_histories = {}

def get_memory(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]
    
# main code block which runs the llm agent
# ... (imports remain the same)

def run_llm_agent(retriever, query, session_id='default', use_rag=True):
    # intialize the memory which stores last 3 prompts
    memory = ConversationBufferWindowMemory(
        memory_key="history",
        chat_memory=get_memory(session_id),
        return_messages=True,
        k=3,
    )

    # creating a prompt template to provide while wrapping so as to provide a prompt in a specified format
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an intelligent and concise AI assistant.
                Your responsibilities include:
                1. Answering questions clearly and accurately.
                2. Performing math when required.
                3. Providing definitions when needed.
                4. Using retrieved context **only if** the user asks something new or not covered in the previous conversation.
                5. If the user asks a follow-up or context-based question, **use the conversation history** first, and only fall back to retrieval if history is not enough.
                Always answer in a helpful and friendly tone. If a question is ambiguous, ask a clarifying question. Never make up facts.
            """),
            ("user", "{input}"),
        ]
    )

    # creating a chain which takes output of prompt template and passes to llm
    chain = prompt_template | llm

    # wrapping the chain with memory 
    wrapped_chain_with_memory = RunnableWithMessageHistory(
        chain,
        lambda session_id=session_id: get_memory(session_id),
        input_messages_key="input",
        history_messages_key="history",
    )

    # DEBUG: Print history before invoking the LLM
    print("\n==== Conversation History ====")
    for msg in memory.chat_memory.messages:
        print(f"{msg.type.upper()}: {msg.content}")
    print("================================\n")

    # DEBUG: Store the original user query in memory
    memory.chat_memory.add_user_message(query)

    # === MEMORY-DEPENDENT BRANCH ===
    # Ask if the current question can be answered using only past conversation (memory)
    memory_check_prompt = (
        "Decide if this question can be answered solely based on the previous conversation without external lookup. "
        "Reply YES or NO.\n\n"
        f"QUESTION: {query}"
    )

    should_memory = wrapped_chain_with_memory.invoke(
        {"input": memory_check_prompt},
        config={"configurable": {"session_id": session_id}},
    ).content.strip()

    # If answerable from memory, just return LLM's response using memory context
    if should_memory.upper() == "YES":
        # Generate answer using memory alone
        memory_only_answer = wrapped_chain_with_memory.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        ).content.strip()

        # DEBUG: Store the AI's memory-based answer in memory (memory branch)
        memory.chat_memory.add_ai_message(memory_only_answer)

        return {
            "answer": memory_only_answer,
            "tool_used": "Tool: Memory-based LLM",
            "context_snippets": [],
        }

    # === CALCULATION BRANCH ===
    # Ask the LLM if this is a calculation
    calc_check_prompt = (
        "Decide if this question requires performing a calculation. "
        "Reply only with YES or NO.\n\n"
        f"QUESTION: {query}"
    )

    should_calc = wrapped_chain_with_memory.invoke(
        {"input": calc_check_prompt},
        config={"configurable": {"session_id": session_id}},
    ).content.strip()

    if should_calc.upper() == "YES":
        # prompt for calculation
        calc_prompt = (
            "Perform this calculation and return the numeric result with the process of how you have solved it:\n"
            "your answering format should be like this: result : {numeric result} \n reasoning : {reasoning on how you have solved it}"
            "Dont return anything except numeric result and reasoning, only ask for something if question in incomplete or more context is required"
            f"{query}"
        )

        result = wrapped_chain_with_memory.invoke(
            {"input": calc_prompt},
            config={"configurable": {"session_id": session_id}},
        ).content.strip()

        # DEBUG: Store the AI's answer in memory
        memory.chat_memory.add_ai_message(result)

        return {
            "answer": result.split('reasoning')[0],
            "tool_used": "Tool: LLM Calculator",
            "context_snippets": [f"Reasoning \n{result.split('reasoning')[1]}"],
        }

    # === DEFINITION BRANCH ===
    if "define" in query.lower() or "definition" in query.lower():
        # DEBUG: Store the original user query in memory
        memory.chat_memory.add_user_message(query)

        # design a prompt for word extraction
        extracted_word_prompt = f"""Extract the single word or phrase that needs to be defined from this question: '{query}'
        Just return the word or phrase without any explanation."""
        
        # ask the LLM to extract the word to be defined from the query
        extracted_word = wrapped_chain_with_memory.invoke(
            {"input": extracted_word_prompt},
            config={"configurable": {"session_id": session_id}},
        ).content.strip()

        # DEBUG: Store the AI's extraction result in memory
        memory.chat_memory.add_ai_message(f"Extracted definition target: {extracted_word}")

        if extracted_word:
            definition = get_definition(extracted_word)
            if definition:
                # adding a definition + llm layer to answer the questions
                definition_context = f"Definition of {extracted_word}: {definition}"

                # definition prompt for answering definitions based on dictionary definition + llm tuning
                definition_prompt = f"{definition_context}\n\nNow answer the user query: '{query}'"

                final_answer = wrapped_chain_with_memory.invoke(
                    {"input": definition_prompt},
                    config={"configurable": {"session_id": session_id}},
                ).content.strip()

                # DEBUG: Store the AI's definition answer in memory
                memory.chat_memory.add_ai_message(final_answer)

                return {
                    "answer": final_answer,
                    "tool_used": "Tool: Dictionary API + LLM Formatter",
                    "context_snippets": [definition_context],
                }
            else:
                # fallback if not found in dictionary
                llm_definition_prompt = f"Define the term '{extracted_word}' and give two definitions if possible."

                answer = wrapped_chain_with_memory.invoke(
                    {"input": llm_definition_prompt},
                    config={"configurable": {"session_id": session_id}},
                ).content.strip()

                # DEBUG: Store the AI's fallback definition in memory
                memory.chat_memory.add_ai_message(answer)

                return {
                    "answer": f"This definition was not present in the dictionary, I will tell you what I know:\n{answer}",
                    "tool_used": "Tool: LLM generated definition",
                    "context_snippets": ["The definition was LLM generated"],
                }

    # === RAG BRANCH (only if user selected “Use sample documents”) ===
    if use_rag and retriever is not None:
        # creating a system prompt for the agent
        system_prompt = (
            "You are a smart assistant with access to the provided context on basis of which you have to answer questions : \n"
            "Context:\n{context}"
            "Be concise. Use at most 4 sentences unless specified.\n\n"
        )

        # defining the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

        # creating a chain of actions for documents
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # combining the document retriever with the question-answer chain
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

        # reuse retriever that was built once
        retrieved_docs = retriever.invoke(query)

        # DEBUG: Store retrieved docs context in memory
        retrieved_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        if retrieved_context.strip():
            from langchain_core.messages import AIMessage
            memory.chat_memory.add_message(AIMessage(content=f"[RAG Context]\n{retrieved_context}"))

        # executing the RAG chain
        response = retrieval_chain.invoke({"input": query})

        # DEBUG: Store the AI's RAG response in memory
        memory.chat_memory.add_ai_message(response["answer"])

        return {
            "answer": response["answer"],
            "tool_used": "Tool: Retrieval Augmented Generation",
            "context_snippets": [doc.page_content for doc in retrieved_docs],
        }

    # === FALLBACK LLM-ONLY ANSWER (no RAG) ===
    # If we get here, we didn’t match calc or definition, and either use_rag=False or no retriever
    fallback_answer = wrapped_chain_with_memory.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    ).content.strip()

    # DEBUG: Store the AI's fallback LLM answer in memory
    memory.chat_memory.add_ai_message(fallback_answer)

    return {
        "answer": fallback_answer,
        "tool_used": "Tool: LLM Only",
        "context_snippets": [],
    }

