from typing import List, Dict, Optional
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
# from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
import chainlit as cl
from adapted_vllm import VLLMOpenAI
from openai import OpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
import os
# from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from chainlit.types import ThreadDict
import chainlit as cl
from langchain_community.document_loaders import ArxivLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, RetrievalQA
from literalai import LiteralClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.utils import filter_complex_metadata
# from langchain_community.chat_message_histories import ChatMessageHistory # adapted class defined below
# from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory, BaseMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
# from chat_adapted import ChatPromptTemplate, MessagesPlaceholder  # adapted class imported
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.llms import VLLMOpenAI
from adapted_vllm import VLLMOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import os
# import uuid
# import chromadb
from chainlit.types import ThreadDict
# from chromadb.config import Settings
from langchain.memory import ConversationBufferMemory
# from langchain_community.chat_message_histories import SQLChatMessageHistory
# from sqlalchemy. ext. asyncio import create_async_engine
from langchain.callbacks.base import BaseCallbackHandler
from pydantic.v1 import BaseModel, Field
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings


load_dotenv()
# client = LiteralClient()


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)
    max_messages: int = 20  # Set the limit 'K' of messages to keep

    def add_message(self, message: BaseMessage) -> None:
        """Add a single message to the store, keeping only the last K messages."""
        self.messages.append(message)
        self.messages = self.messages[-self.max_messages:]

    def clear(self) -> None:
        self.messages = []


namespaces = set()

model = VLLMOpenAI(
    openai_api_key=os.getenv("VLLM_API_KEY"),
    openai_api_base=os.getenv("VLLM_ENDPOINT"),
    model_name="amrachraf/arxiv-assistant-merged_peft_model",
    extra_body={"SamplingParams": {"min_tokens": 1,
                                   "skip_special_tokens": True,
                                   }, "trust_remote_code": True},
    temperature=0,
    max_tokens=4000,
)

# embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

# embeddings_model = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("HF"),
#                                                      model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings_model = HuggingFaceEndpointEmbeddings(
    model=os.getenv("EMBED_ENDPOINT"), 
    huggingfacehub_api_token=os.getenv("HF")
)


@cl.oauth_callback
def oauth_callback(
        provider_id: str,
        token: str,
        raw_user_data: Dict[str, str],
        default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


# @cl.password_auth_callback
# def auth():
#     return cl.User(identifier="test")


openai_api_base = os.getenv("VLLM_ENDPOINT") 

openai_client = OpenAI(
    api_key=os.getenv("VLLM_API_KEY"),
    base_url=openai_api_base,
)


def process_arxiv_pdfs(arxiv_query: str):
    search_sys_prompt = f"""You identify a topic name from a user query or sentence. And reply with it only without any additional notes or text.
You always reply only with a topic name of maximum 3 words and minimum 1 word.
Extract the main topic from the user query including the date if any.

If no topic can be extracted, reply with the same user input unmodified. generate your response to be only the original user query.

Examples in form of [query] 'reply' .. don't include the brackets nor quotes in your reply and don't include any additional notes inside parentheses in your reply:
- [What's Atom?] 'Atom'
- [NLP papers published in 2024] 'NLP papers 2024'
- [Attention is all you need] 'Attention is all you need'
- [Object Counting: You Only Need to Look at One] 'Object Counting: You Only Need to Look at One'
- [What is the best way to train a model?] 'best way to train a model'
- [Explain regression models] 'regression models'
- [Tell me about the latest research in AI] 'latest research AI'

Ensure that the output is only the main topic and date if any, without adding any extra words, details, explanations, notes or formatting.
Always follow all the mentioned instructions and the same format and reply with the topic name only. don't complete anything missing from the query, just reply with the topic name only without specifying that this is the reply or the topic, also dont specify that there is no date provided.
Eliminate any notes inside parentheses from your response."""

    chat_response = openai_client.chat.completions.create(
        model="amrachraf/arxiv-assistant-merged_peft_model",
        temperature=0,
        max_tokens=1000,
        messages=[
            {"role": "user", "content": search_sys_prompt + "\n" + arxiv_query}
        ],
        extra_body={
            "min_tokens": 1,
            "skip_special_tokens": True,
        }
    )

    arxiv_query = chat_response.choices[0].message.content
    arxiv_query = arxiv_query.strip().replace("\n", "")
    arxiv_docs = ArxivLoader(query=arxiv_query, load_max_docs=3, top_k_results=3,
                             load_all_available_meta=True,
                             doc_content_chars_max=os.getenv("MAX_ARXIV_CHAR")).load()

    cl.user_session.set("arxiv_docs", arxiv_docs)
    cl.user_session.set("arxiv_query", arxiv_query)

    return arxiv_docs, arxiv_query


def create_vectorstore(docs: List[Document], collection_name: str):
    text_splitter = SemanticChunker(
        embeddings_model,
        breakpoint_threshold_amount=0.8
        # breakpoint_threshold_type="percentile"
    )

    arxiv_docs = filter_complex_metadata(docs)

    texts = text_splitter.split_documents(arxiv_docs)

    for i, text in enumerate(texts):
        text.metadata["source"] = f"source_{i}"

    cl.user_session.set("docs", texts)

    vectorstore = Chroma.from_documents(texts, embeddings_model, collection_name=collection_name, persist_directory="./chroma")

    namespace = collection_name

    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    namespaces.add(namespace)

    index_result = index(
        texts,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return vectorstore


@cl.step(type="tool", name="Arxiv API")
async def gen_query(human_query: str):
    arxiv_docs, arxiv_query = process_arxiv_pdfs(human_query)
    doc_len = len(arxiv_docs)
    await cl.Message(content=f"Retrieved {doc_len} documents for `{arxiv_query}` Processing results.").send()
    return arxiv_docs


@cl.step(type="embedding", name="Generate Vectorstore")
async def process_query(arxiv_docs: List[Document]):
    arxiv_papers = [
        f"""__Published:__ {doc.metadata['Published']}s
__Title:__ {doc.metadata['Title']}
__Authors:__ {doc.metadata['Authors']}
__Summary:__ {doc.metadata['Summary']}
__URL:__ {doc.metadata['links'][-1]}

""" for doc in arxiv_docs]
    await cl.Message(content=f"Processing and chunking retrieved articles. This might take sometime.").send()

    vectorstore = await cl.make_async(create_vectorstore)(arxiv_docs, collection_name=cl.context.session.thread_id)

    arxiv_papers_msg = cl.Message(content="".join(arxiv_papers))
    arxiv_papers_content = arxiv_papers_msg.content

    return vectorstore, arxiv_papers_content


@cl.step(type="run", name="Arxiv Assistant Chain generation")
async def chain(human_query: str, memory: Dict[str, BaseChatMessageHistory]):
    arxiv_docs = await gen_query(human_query)
    vectorstore, arxiv_papers_content = await process_query(arxiv_docs)

    retriever = vectorstore.as_retriever(search_type="mmr",
                                         search_kwargs={"k": 2,
                                                        "fetch_k": 50
                                                        }
                                         )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an arXiv assistant for question-answering tasks. "
        "your name is Marvin, and you're developed by Amr Achraf. "
        "you provide detailed, comprehensive and helpful responses to any request, specially requests related to scientific papers published on arXiv. "
        "structure your responses and reply in a clear scientific manner. "
        "ensure to greet the user at the start of the first message of the conversation only. "
        "ensure to ask the user if your response was clear and sufficient and if he needs any other help. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. keep the "
        "answer concise and structured."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in memory:
            memory[session_id] = ChatMessageHistory()
            memory[session_id].add_ai_message(arxiv_papers_content) # adding summary of retrieved papers to chat history in first invokation
        return memory[session_id]

    runnable = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return runnable, arxiv_papers_content


@cl.step(name="Prompt user message")
async def send_message_to_user(human_query: str):
    await cl.Message(content=f"Database for `{human_query}` is created successfully. You can ask questions.").send()


@cl.on_chat_start
async def on_chat_start():
    store = {}

    cl.user_session.set("memory", store)

    memory = cl.user_session.get("memory")

    app_user = cl.user_session.get("user")

    greeting_msg = cl.Message(f"Hello {app_user.identifier}")
    await greeting_msg.send()

    query = None

    while query is None:
        query = await cl.AskUserMessage(content="Please enter a topic to begin!", timeout=15).send()
    human_query = query['output']

    runnable, arxiv_papers_content = await chain(human_query, memory)

    await cl.Message(content=arxiv_papers_content).send()

    await send_message_to_user(human_query)

    cl.user_session.set("runnable", runnable)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    store = {}

    store[cl.user_session.get("id")] = ChatMessageHistory()

    memory = store[cl.user_session.get("id")]
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.add_user_message(message["output"])
        else:
            memory.add_ai_message(message["output"])

    cl.user_session.set("memory", store)

    memory = cl.user_session.get("memory")

    # print("memory", memory)  # ### checking memory rebuilt

    collection_name = cl.context.session.thread_id

    vectorstore = Chroma(embedding_function=embeddings_model, collection_name=collection_name, persist_directory="./chroma")

    retriever = vectorstore.as_retriever(search_type="mmr",
                                         search_kwargs={
                                             "k": 2,
                                             "fetch_k": 50
                                         }
                                         )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an arXiv assistant for question-answering tasks. "
        "your name is Marvin, and you're developed by Amr Achraf. "
        "you provide detailed, comprehensive and helpful responses to any request, specially requests related to scientific papers published on arXiv. "
        "structure your responses and reply in a clear scientific manner. "
        "ensure to greet the user at the start of the first message of the conversation only. "
        "ensure to ask the user if your response was clear and sufficient and if he needs any other help. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. keep the "
        "answer concise and structured."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in memory:
            memory[session_id] = ChatMessageHistory()
        return memory[session_id]

    runnable = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set() 

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['entry_id'], d.metadata['Title'])
                self.sources.add(source_page_pair) 

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n\n".join([f"{source}\nentry_id: {entry_id}\ntitle: {title}" for source, entry_id, title in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    result = await runnable.ainvoke(
        {"input": message.content},
        config={
            "configurable": {"session_id": cl.user_session.get("id")},
            'callbacks': [cl.LangchainCallbackHandler(), PostMessageHandler(msg)]
        }, 
    )

    await msg.send()
    await cl.Message(content="".join(result['answer']), type="assistant_message").send()

