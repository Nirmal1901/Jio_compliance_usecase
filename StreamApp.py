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
import pandas as pd


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define weights
WEIGHT_MAP = {
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5
}

# Predefined regulatory questions
BASE_REGULATORY_QUESTIONS = {
    "Loan Agreement": [
        {"requirement": "As complaints received against NBFCs generally pertain to charging of high interest/penal charges, NBFCs shall mention the penalties charged for late repayment in bold in the loan agreement.", "weight": "high"},
        {"requirement": "Penalty, if charged, for non-compliance of material terms and conditions of loan contract by the borrower shall be treated as ‘penal charges’. Material terms and conditions to be defined in the loan agreement. Penal charges to be defined in a Board approved policy.", "weight": "medium"},
        {"requirement": "The quantum and reason for penal charges shall be clearly disclosed to the customers in the loan agreement and most important terms & conditions/Key Fact Statement (KFS).", "weight": "medium"},
        {"requirement": "Ensure that changes in interest rates and charges are affected only prospectively. A suitable condition in this regard must be incorporated in the loan agreement.", "weight": "high"},
        {"requirement": "Decision to recall/accelerate payment or performance under the agreement shall be in consonance with the loan agreement.", "weight": "low"},
        {"requirement": "Release all securities on repayment of all dues or on realisation of the outstanding amount of loan subject to any legitimate right or lien for any other claim they may have against the borrower. If such right of set off is to be exercised, the borrower shall be given notice about the same with full particulars about the remaining claims and the conditions. (Suitable clause to be added in the loan agreement).", "weight": "medium"},
        {"requirement": "SMA/NPA classification along with the exact due dates for repayment of a loan, frequency of repayment, breakup between principal, interest and examples of SMA/NPA classifications should be specifically mentioned in the Loan agreement.", "weight": "medium"},
        {"requirement": "Interest Rate Model: Adopt an interest rate model taking into account relevant factors such as cost of funds, margin and risk premium and determine the rate of interest to be charged for loans and advances. The rate of interest and the approach for gradations of risk and rationale for charging different rate of interest to different categories of borrowers shall be disclosed to the borrower or customer in the application form and communicated explicitly in the sanction letter. The rate of interest must be annualised rate so that the borrower is aware of the exact rates that would be charged to the account.", "weight": "high"},
        {"requirement": "Annual Percentage Rate (APR) shall be disclosed upfront and shall also be a part of the Key Fact Statement.", "weight": "high"},
        {"requirement": "The possible impact of change in benchmark interest rate on the loan leading to changes in EMI and/or tenor or both.", "weight": "medium"},
        {"requirement": "The loan agreement with the borrower shall contain clauses for conduct of audit (as defined in ) at the behest of lender(s). In cases where the audit report submitted remains inconclusive or is delayed due to non-cooperation by the borrower, Applicable NBFCs shall conclude on status of the account as a fraud or otherwise based on the material available on their record and their own internal investigation / assessment in such cases.(Per MD on Fraud Risk Management in NBFC).", "weight": "medium"},
        {"requirement": "Treatment of Wilful Defaulter and Large Defaulter: Incorporation of covenant: (i) The lender shall incorporate a covenant in the agreement while extending credit facility to a borrower that it shall not induct a person whose name appears in the LWD on its board or as a person in charge and responsible for the management of the affairs of the entity. (ii) In case such a person is found to be on its board or as a person in charge and responsible for the management of the affairs of the entity, the borrower would take expeditious and effective steps for removal of such a person from the board or from being in charge of its management. (iii) Under no circumstances shall a lender renew/ enhance/ provide fresh credit facilities or restructure existing facilities provided to such a borrower so long as the name of its promoter and/or the director (s) and/or the person in charge and responsible for the management of the affairs of the entity remains in the LWD.", "weight": "high"},
        {"requirement": "Repossession of vehicles financed: Must have a built-in re-possession clause in the contract/loan agreement with the borrower which must be legally enforceable. To ensure transparency, the terms and conditions of the contract/loan agreement shall also contain provisions regarding: (i) Notice period before taking possession; (ii) Circumstances under which the notice period can be waived; (iii) The procedure for taking possession of the security; (iv) A provision regarding final chance to be given to the borrower for repayment of loan before the sale/ auction of the property; (v) The procedure for giving repossession to the borrower; and (vi) The procedure for sale/auction of the property.", "weight": "medium"},
        {"requirement": "Repossession of vehicles financed: A copy of such terms and conditions must be made available to the borrower. NBFCs shall invariably furnish a copy of the loan agreement along with a copy each of all enclosures quoted in the loan agreement to all the borrowers at the time of sanction/ disbursement of loans, which forms a key component of such contracts/ loan agreements.", "weight": "medium"},
        {"requirement": "Details of grievance redressal officer (GRO).", "weight": "low"},
        {"requirement": "Customer consent for transactional/promotional SMS or Voice call (a suitable clause to be added in T & C).", "weight": "low"}
    ],
    "KFS": [
        {"requirement": "Ensure KFS is provided with a unique proposal number.", "weight": "medium"},
        {"requirement": "Include data fields as per KFS format: Type of loan, Sanctioned Loan, Disbursal Schedule, Installment details, Interest rate (%) and type (fixed or floating or hybrid), Additional Information in case of Floating rate of interest, Fee/Charges, Annual Percentage Rate (APR), Details of Contingent Charges and Part 2 (Other qualitative information).", "weight": "high"},
        {"requirement": "The quantum and reason for penal charges shall be clearly disclosed to the customers in the loan agreement and most important terms & conditions/Key Fact Statement (KFS).", "weight": "high"},
        {"requirement": "Annual Percentage Rate (APR) shall be disclosed upfront and shall also be a part of the Key Fact Statement.", "weight": "high"},
        {"requirement": "Cooling Off Period to be defined.", "weight": "medium"},
        {"requirement": "Penal charges to be mentioned in Bold.", "weight": "high"},
        {"requirement": "Repayment Schedule has to be provided to the borrower along with the dates of the repayment.", "weight": "medium"}
    ]
}
class GraphState(TypedDict):
    question: str
    context: str
    docs: list
    response: str
    retriever: object

def load_and_split_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def store_embeddings(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    qdrant = Qdrant.from_documents(
        chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="compliance_chunks"
    )
    return qdrant

def retrieve_node(state: GraphState):
    docs = state["retriever"].similarity_search(state["question"], k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {**state, "docs": docs, "context": context}



def verify_loan_document(file_path: str) -> bool:
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Extract first 2–3 pages or chunks (adjust based on chunk length)
    sample_texts = [page.page_content for page in pages[:3]]
    joined_text = "\n\n".join(sample_texts)

    prompt_template = """
You are an expert legal document classifier.

Your task is to read the given document content and determine if it is a **Loan Agreement**.

Here is the content:

---
{content}
---

Does this document appear to be a **Loan Agreement**?

Answer only YES or NO.
"""

    prompt = PromptTemplate.from_template(prompt_template)
    llm = Ollama(model="llama3:8b", temperature=0)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"content": joined_text}).strip().lower()
    return "yes" in response


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

def calculate_confidence(results):
    total_score = 0
    max_score = 0
    for r in results:
        weight = WEIGHT_MAP.get(r["Weight"], 1.0)
        max_score += weight
        compliance = r["Compliance"].strip().lower()
        if "✅" in compliance:
            total_score += weight
        elif "⚠️" in compliance:
            total_score += weight * 0.5
    confidence_percent = (total_score / max_score) * 100 if max_score else 0
    return round(confidence_percent, 2)

def process_files(agreement_file, questions):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(agreement_file.getvalue())
        temp_pdf_path = temp_pdf.name

    

    



    try:

        is_valid = verify_loan_document(temp_pdf_path)
        if not is_valid:
            raise ValueError("The uploaded document does not appear to be a Loan Agreement.")


        chunks = load_and_split_pdf(temp_pdf_path)
        logger.info(f"🔹 Loaded and split PDF into {len(chunks)} chunks")

        qdrant = store_embeddings(chunks)
        logger.info(f"🔹 Chunks embedded and indexed in Qdrant")

        builder = StateGraph(GraphState)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        graph = builder.compile()

        results = []
        for item in questions:
            requirement = item["requirement"]
            weight = item["weight"]
            logger.info(f"🔍 Processing: {requirement}")
            result = graph.invoke({"question": requirement, "retriever": qdrant})
            resp = result["response"]
            first_line = resp.splitlines()[0].strip()
            reason = "\n".join(resp.splitlines()[1:]).strip()
            context = result["context"]
            results.append({
                "Requirement": requirement,
                "Compliance": first_line,
                "Reason": reason,
                "Chunks": context,
                "Weight": weight
            })

        confidence = calculate_confidence(results)
        return results, confidence

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise e
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

def main():
    st.set_page_config(page_title="Compliance Validator", layout="wide")

    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stMarkdown {
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📑 Loan Document Compliance Checker")

    with st.sidebar:
        st.header("📁 Settings")
        doc_type = st.selectbox("Select Document Type", list(BASE_REGULATORY_QUESTIONS.keys()))
        default_questions = BASE_REGULATORY_QUESTIONS.get(doc_type, [])

        st.markdown("### 🛠️ Weight Presets")
        selected_bulk_weight = st.selectbox("Set All Weights To", ["Do not change", "high", "medium", "low"])
        if selected_bulk_weight != "Do not change" and "reg_df" in st.session_state:
            st.session_state.reg_df.loc[st.session_state.reg_df["use"], "weight"] = selected_bulk_weight

        st.markdown("### 🗂️ Requirement Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select All"):
                if "reg_df" in st.session_state:
                    st.session_state.reg_df["use"] = True
        with col2:
            if st.button("🚫 Deselect All"):
                if "reg_df" in st.session_state:
                    st.session_state.reg_df["use"] = False

    # Initialize editable DataFrame
    # Initialize or update regulatory requirements table if doc_type changes
    if "selected_doc_type" not in st.session_state or st.session_state.selected_doc_type != doc_type:
        st.session_state.selected_doc_type = doc_type
        st.session_state.reg_df = pd.DataFrame(default_questions)
        st.session_state.reg_df["use"] = True


    st.markdown("## 📜 Regulatory Requirements")
    with st.expander("📝 Review / Edit Requirements", expanded=False):
        edited_df = st.data_editor(
            st.session_state.reg_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "requirement": st.column_config.TextColumn("Requirement"),
                "weight": st.column_config.SelectboxColumn("Weight", options=["high", "medium", "low"]),
                "use": st.column_config.CheckboxColumn("Use")
            }
        )
        st.session_state.reg_df = edited_df  # Update session state

    editable_questions = edited_df[edited_df["use"]].to_dict(orient="records")

    st.markdown("---")
    st.markdown("## 📤 Upload Loan Agreement")

    agreement_file = st.file_uploader("Upload your loan agreement PDF", type=["pdf"], help="Only PDF format is supported.")

    st.markdown("---")
    st.markdown("## ✅ Compliance Evaluation")


    if st.button("🚀 Run Compliance Check"):
        if not agreement_file:
            st.error("🚫 Please upload the document first.")
            return
        if not editable_questions:
            st.error("🚫 No regulatory requirements selected.")
            return

        with st.spinner("🔄 Processing document and evaluating compliance..."):
            try:
                results, confidence = process_files(agreement_file, editable_questions)
                df = pd.DataFrame(results)

                st.success("✅ Compliance Evaluation Complete")

                # Use a container for proper spacing
                with st.container():
                    st.markdown("## 📊 Results Overview")

                    # Use columns for metric and button
                    metric_col, button_col = st.columns([2, 1])
                    with metric_col:
                        st.metric("📈 Overall Compliance Score", f"{confidence}%")

                    with button_col:
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Download Results", csv, "compliance_results.csv", "text/csv")

                    st.markdown("---")
                    with st.expander("📄 View Detailed Results", expanded=True):
                        st.dataframe(df, use_container_width=True)

            except Exception as e:

                if "Loan Agreement" in str(e):
                    st.error("❌ This doesn't appear to be a valid Loan Agreement. Please upload the correct document.")
                else:
                    st.error(f"❌ An error occurred: {e}")

                



if __name__ == "__main__":
    main()
