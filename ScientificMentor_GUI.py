import os
from dotenv import load_dotenv, dotenv_values
import os
import nest_asyncio  

nest_asyncio.apply()

load_dotenv(dotenv_path="/your_dotenv_path/.env")
os.environ['USER_AGENT'] = 'myagent'
local_llm = "llama3.1"  # 'gemma2', 'mistral', 'llama3.1'
from langchain_openai import ChatOpenAI
import json
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents.agent_types import AgentType
from langchain_community.document_loaders import DirectoryLoader

# Initialize Language Model and Load Content
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import re

# Set up paths and initialize models
pdf_folder_path = 'your_pdf_folder_path'
embeddings_model = SentenceTransformerEmbeddings(
    model_name='intfloat/e5-large-v2') 
# Initialize Chroma storage
chroma_db_path = 'your_chroma_db_path'

##Rebuild database
def load_pdfs_from_folder(folder_path):
    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            pdf_texts.append(extract_text_from_pdf(file_path))
    return pdf_texts

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text_with_splitter(text, max_chunk_size=2000):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


# # Process PDFs and prepare chunks
pdf_texts = load_pdfs_from_folder(pdf_folder_path)
all_documents = []

for text in pdf_texts:
    chunks = chunk_text_with_splitter(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    all_documents.extend(documents)

# Store all chunks and embeddings in Chroma storage
vectorstore = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings_model,
    persist_directory=chroma_db_path
)

# Persist Chroma storage
vectorstore.persist()


# Load vectorstore

vectorstore = Chroma(
    embedding_function=embeddings_model,  # Use the same embedding model as in creation
    persist_directory=chroma_db_path,

)

#
# Use as a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Retrieval Grader

# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt_retrieval_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing the relevance
    of a retrieved document to a user question. Grade it as relevant if the document contains keywords related to the user question.
    Provide a simple binary "yes" or "no" score as JSON with the single key "score" and no additional text.\n
    Here is the retrieved document:\n\n{document}\n\n
    Here is the user question:\n{question}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {{"score": "yes" or "no"}}
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt_retrieval_grader | llm | JsonOutputParser()

# Generator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt_generator = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to provide a detailed and critical response for the question.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context: {Context}
    Question: {Question}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["Context", "Question"],
)

llm = ChatOllama(model=local_llm, temperature=0.05)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt_generator | llm | StrOutputParser()

# Hallucination Grader and Answer Grader

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt_hallucination_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing if the answer is grounded in the provided facts.
    Return one of the following as a JSON response:
    - {{"score": "yes"}}
    - {{"score": "no"}}
    Do not write anything else.

    Facts: {documents}
    Answer: {generation}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt_hallucination_grader | llm | JsonOutputParser()

### Answer Grader
# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)
# Prompt
prompt_answer_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON object with a single key 'score' (without quotes around the key) and no preamble or explanation.
    The response should strictly follow the format: {{"score": "yes"}} or {{"score": "no"}}.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt_answer_grader | llm | JsonOutputParser()

###Router
# from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt_question_router = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
    user question to a vectorstore or web search. Use the 'vectorstore' for questions on droplet-based microfluidics.\n
    You do not need to be stringent with the keywords\n
    in the question related to these topics. Otherwise, use web-search. Give a  choice 'web_search'\n
    or 'vectorstore'. Return the a JSON with a single key 'datasource' and \n
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt_question_router | llm | JsonOutputParser()

###Web Search

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)


####Control Flow
from pprint import pprint
from typing import List

import time

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph


# State


# Define State
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    # counter_or_generate: str
    documents: List[str]
    hallu_count: int
    answer_count: int
    retrieve_count: int


# Define Functions
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

compressor = CohereRerank(
    
    model="rerank-english-v3.0"
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


def retrieve(state):
    """Retrieve documents from vectorstore."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = compression_retriever.get_relevant_documents(question)  # retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """Generate answer using RAG on retrieved documents."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"Context": documents, "Question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """Determine document relevance and flag for web search if needed."""
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if "score" in score:
            if score["score"].lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """Perform web search based on the question."""
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})


    if isinstance(docs[0], dict) and "content" in docs[0]:
        web_results = "\n".join([d["content"] for d in docs])
    else:
        web_results = "\n".join(docs)

    web_results = Document(page_content=web_results)

    # If documents already exist, append to them, else initialize with web results
    if "documents" not in state or state["documents"] is None:
        state["documents"] = [web_results]
    else:
        state["documents"].append(web_results)

    return {"documents": state["documents"], "question": question}


def route_question(state):
    """Route the question to either vectorstore or web search"""
    print("---ROUTE QUESTION---")
    source = question_router.invoke({"question": state["question"]})
    print("source: ", source)

    if "datasource" in source:
        print("source[datasource]:", source["datasource"])
        if source["datasource"] == "web_search":
            return "websearch"
        return "vectorstore"
    return "websearch"


def decide_to_generate(state):
    """Determine if we should generate or add web search results."""
    print("---DECIDE TO GENERATE---")
    if state["web_search"] == "Yes":
        return "websearch"
    return "generate"


# Build Graph
def counter_answer(state):
    answer_count = state["answer_count"]
    answer_count = answer_count + 1
    return {"answer_count": answer_count}


def counter_hallu(state):
    hallu_count = state["hallu_count"]
    hallu_count = hallu_count + 1
    return {"hallu_count": hallu_count}


def counter_retrieve(state):
    retrieve_count = state["retrieve_count"]
    retrieve_count = retrieve_count + 1
    return {"retrieve_count": retrieve_count}


def grade_generation_v_documents_and_question(state):
    """Grade the generation against the retrieved documents and the question."""
    print("---GRADE GENERATION---")
    print("generation:\n", state["generation"])
    print("documents:\n", format_docs(state["documents"]))
    answer_count = state["answer_count"]
    hallu_count = state["hallu_count"]
    retrieve_count = state["retrieve_count"]

    print("answer_count: ", answer_count)
    print("hallu_count: ", hallu_count)
    print("retrieve_count: ", retrieve_count)

    # Grade the generation based on hallucination
    score = hallucination_grader.invoke(
        {"generation": state["generation"], "documents": format_docs(state["documents"])})
    if hallu_count == 2:
        score["score"] = "yes"
    else:
        score = hallucination_grader.invoke(
            {"generation": state["generation"], "documents": format_docs(state["documents"])})

    # Debug: Print the entire response to check what it contains
    print("Hallucination Grader Response:", score)
    # Check if "score" exists in the response
    if "score" in score:
        if score["score"] == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

            # Check if the answer addresses the question
            score_ = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
            if answer_count == 2:
                score_["score"] = "yes"
            else:
                score_ = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
            if "score" in score_ and score_["score"] == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    else:
        # Handle the case where "score" is not present
        print("---ERROR: 'score' NOT FOUND IN RESPONSE---")
        # Log the response to help debug
        print("Full Hallucination Grader Response:", score)
        if retrieve_count == 2:
            return "useful"
        return "back retrieve"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

workflow.add_node("counter_answer", counter_answer)  # counter_answer
workflow.add_node("counter_hallu", counter_hallu)  # counter_hallu
workflow.add_node("counter_retrieve", counter_retrieve)  # counter_retrieve

# Entry point and conditional routing
workflow.set_conditional_entry_point(route_question, {"websearch": "websearch", "vectorstore": "retrieve"})

# Define edges
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate,
                               {"websearch": "websearch", "generate": "generate"})

workflow.add_edge("counter_answer", "websearch")
workflow.add_edge("websearch", "generate")
workflow.add_edge("counter_hallu", "generate")
workflow.add_edge("counter_retrieve", "retrieve")

# Grade generation
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {"not supported": "counter_hallu", "useful": END, "not useful": "counter_answer",
     "back retrieve": "counter_retrieve"}
)



# Compile and generate graph
app = workflow.compile()

import streamlit as st

# -- Streamlit UI ------------------------------------------------------------


st.set_page_config(page_title="DropMicroFluidAgents🧠🔎🌐🧑‍💻 Scientific Mentor", layout="wide")
st.title(" DropMicroFluidAgents 🧠🔎🌐🧑‍💻Scientific Mentor")
st.markdown("Powered by **LLM-agents**")
query = st.text_input("Enter your question:", placeholder="e.g. How does droplet size affect trapping efficiency?")

if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running..."):
            # Initialize state
            init_state = GraphState(
                question=query,
                documents=[],
                generation="",
                web_search="No",
                hallu_count=0,
                answer_count=0,
                retrieve_count=0
            )
            # Execute
            result = app.invoke(init_state)
        # Display
        st.subheader("Answer")
        st.write(result.get("generation", "No answer generated."))



