import os
import csv
import json
import logging
import tempfile
import requests
import pandas as pd
import streamlit as st
from typing import List, TypedDict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State for LangGraph
class GraphState(TypedDict):
    question: str
    context: str
    docs: list
    response: str
    retriever: object

# Load questions from JSON
def load_questions(filepath: str) -> List[dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

# Load and split PDF
def load_and_split_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

# Store chunks in Qdrant
def store_embeddings(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    qdrant = Qdrant.from_documents(
        chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="compliance_chunks"
    )
    return qdrant

# Node: retrieve
def retrieve_node(state: GraphState):
    docs = state["retriever"].similarity_search(state["question"], k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {**state, "docs": docs, "context": context}

# Node: generate response
def generate_node(state: GraphState):
    llm = Ollama(model="llama3:8b", temperature=0)
    template = """
You are a compliance validation expert.

Requirement: {question}

Below is the extracted content from a loan agreement relevant to this requirement:

{context}

Please analyze if the context fully satisfies the compliance requirement. Be concise but explain your judgment clearly.

Answer Format:
Compliance: ✅ Compliant / ❌ Non-Compliant / ⚠️ Partially Compliant
Reason: [Brief reasoning]
"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"], "context": state["context"]})
    return {**state, "response": result}

def process_files(agreement_file, questions_file):
    """Process uploaded files and return compliance results"""
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(agreement_file.getvalue())
        temp_pdf_path = temp_pdf.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_json:
        temp_json.write(questions_file.getvalue())
        temp_json_path = temp_json.name

    try:
        # Step 1: Load & chunk PDF
        chunks = load_and_split_pdf(temp_pdf_path)
        logger.info(f"🔹 Loaded and split PDF into {len(chunks)} chunks")

        # Step 2: Embed in Qdrant
        qdrant = store_embeddings(chunks)
        logger.info(f"🔹 Chunks embedded and indexed in Qdrant")

        # Step 3: Load questions
        questions_raw = load_questions(temp_json_path)
        if isinstance(questions_raw, dict) and "questions" in questions_raw:
            questions = questions_raw["questions"]
        else:
            questions = questions_raw

        # Step 4: Setup LangGraph
        builder = StateGraph(GraphState)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        graph = builder.compile()

        # Step 5: Process each question
        results = []
        for q in questions:
            if isinstance(q, str):
                requirement = q
                keywords = []
            elif isinstance(q, dict):
                requirement = q.get("requirement", "")
                keywords = q.get("keywords", [])
            else:
                continue

            logger.info(f"🔍 Processing: {requirement}")
            result = graph.invoke({"question": requirement, "retriever": qdrant})

            # Extract parts
            resp = result["response"]
            first_line = resp.splitlines()[0].strip()
            reason = "\n".join(resp.splitlines()[1:]).strip()
            context = result["context"]

            results.append({
                "Requirement": requirement,
                "Compliance": first_line,
                "Response": resp,
                "Reason": reason,
                "Chunks": context
            })

        return results

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise e
    finally:
        # Clean up temporary files
        for path in [temp_pdf_path, temp_json_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except:
                pass

# Streamlit UI
def main():
    # Page config
    st.set_page_config(page_title="Compliance Validator", layout="wide")

    # Title
    st.title("Loan Agreement Compliance Validator")

    # Document type selection
    doc_type = st.selectbox(
        "Select Document Type",
        ["Loan Agreement", "KFS", "Application Form", "Sanction Letter"],
        index=0
    )

    # File upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Upload {doc_type}")
        agreement_file = st.file_uploader(
            "Upload PDF file", 
            type=["pdf"],
            key="agreement"
        )
    
    with col2:
        st.subheader("Upload Regulatory Requirements")
        questions_file = st.file_uploader(
            "Upload JSON file", 
            type=["json"],
            key="questions"
        )

    # Evaluate button
    if st.button("Evaluate Compliance", type="primary"):
        if agreement_file is None or questions_file is None:
            st.error("Please upload both files before evaluating")
            return

        with st.spinner("Processing documents..."):
            try:
                results = process_files(agreement_file, questions_file)
                
                if results:
                    st.success("Evaluation completed successfully!")
                    
                    # Convert results to DataFrame
                    df = pd.DataFrame(results)
                    
                    # Show results table
                    st.subheader("Compliance Results")
                    st.dataframe(df)
                    
                    # Create download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="compliance_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No results were generated")
            
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()