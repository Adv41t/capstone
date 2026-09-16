# Capstone

The **Capstone-Healthcare Auditing Engine** is an automated compliance and billing audit platform designed to analyze, cross-reference, and validate medical bills against patient referrals and legal reimbursement contracts. 

By leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), the engine ensures patient details match, verifies insurance application, confirms that billed tests correspond to authorized referrals, checks pricing limits against contracted fee schedules, and validates mathematical calculations.

---

## Architecture & Core Modules

The project is structured into modular components:

*   **`app.py`**: The main entry point. Built with **Streamlit**, it provides an interactive web dashboard for uploading documents, initiating audits, displaying checklist results, viewing historical audits, and querying the system.
*   **`checklist.py` (`AuditChecklistEngine`)**: Implements the core 6-point audit verification logic, coordinating between document extractions, database history, and contract RAG.
*   **`extractor.py` (`AuditExtractor`)**: Uses the **Gemini API** with strict Pydantic schema enforcement to extract structured data (patient names, dates of birth, test codes, line items) from raw document text.
*   **`rag_engine.py` (`ContractRAGEngine`)**: Indexes legal contracts (like master service/reimbursement agreements) using **ChromaDB** and **Sentence-Transformers** (`all-MiniLM-L6-v2`) for semantic search of pricing caps and disallowed fee clauses.
*   **`database.py`**: Manages patient records and audit ticket history using **SQLite**, supporting patient fuzzy matching and historical compliance tracking.
*   **`file_reader.py`**: Handles text extraction from uploaded PDF documents using **PyMuPDF (Fitz)**.

---

## The 6-Point Audit Checklist

The engine executes a structured 6-point workflow and compliance checklist:

1.  **Files Fetched**: Ensures patient referral and invoice PDF documents are successfully ingested and parsed into JSON structures.
2.  **Patient Matched**: Cross-references patient names and dates of birth between documents using LLM fuzzy matching (to handle abbreviations or middle name discrepancies).
3.  **Insurance Applied**: Inspects the invoice text to verify if insurance details, copays, or discounts were applied.
4.  **Test Matched**: Verifies if the medical procedure billed matches the exact test authorized in the patient referral.
5.  **Allowed Fees and Pricing Cap**: Uses semantic search to locate the corresponding rate schedule or cap in the legal contract and verifies that the billed amount does not exceed the allowed limit.
6.  **Total Calculation Validation**: Re-calculates the sum of all individual billing line items (including adjustments and write-offs) to verify mathematical correctness.

---

## Technical Stack

*   **Frontend**: Streamlit
*   **LLM API**: Google Gemini (`gemini-2.5-flash`)
*   **Vector Database**: ChromaDB
*   **Embeddings Model**: Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **PDF Processing**: PyMuPDF (`pymupdf`)
*   **Database**: SQLite
*   **Data Validation**: Pydantic

---

## Setup & Execution

### 1. Install Dependencies
Make sure you have python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
This will start the local development server and open the web dashboard in your default browser.
