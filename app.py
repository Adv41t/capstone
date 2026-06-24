import os
import json
import streamlit as st
import google.api_core.exceptions
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

def is_rate_limit_error(exception):
    if isinstance(exception, google.api_core.exceptions.ResourceExhausted):
        return True
    err_str = str(exception).lower()
    return "429" in err_str or "quota" in err_str or "rate limit" in err_str or "resourceexhausted" in err_str

@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_random_exponential(min=10, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def generate_content_with_retry(model, *args, **kwargs):
    return model.generate_content(*args, **kwargs)


# Import our modular components
import database
import file_reader
from extractor import AuditExtractor
from rag_engine import ContractRAGEngine
from checklist import AuditChecklistEngine

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Healthcare Auditing Engine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Gradient headers */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    /* Section Cards */
    .section-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #1e3c72;
        margin-bottom: 1.5rem;
    }
    .check-card-pass {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #2e7d32;
        margin-bottom: 1rem;
    }
    .check-card-fail {
        background-color: #ffebee;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #c62828;
        margin-bottom: 1rem;
    }
    .check-title {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
    }
    /* Status pills */
    .status-pill-pass {
        background-color: #2e7d32;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-pill-fail {
        background-color: #c62828;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Generate a default contract PDF in the backend if not present
STATIC_CONTRACT_PATH = "contract.pdf"

def generate_default_contract(filepath: str) -> None:
    """Generates a default mock contract PDF if it doesn't exist."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    
    contract_text = """
    MASTER MEDICAL SERVICES AGREEMENT & REIMBURSEMENT POLICY
    
    This document outlines the allowed billing fees and pricing caps for medical procedures.
    
    SECTION 1: FEE SCHEDULE & PRICING CAPS
    1.1 Brain MRI (procedure code 70551): The maximum allowed reimbursement rate for a Brain MRI is $450.00. 
    1.2 Chest X-Ray (procedure code 71045): The maximum allowed reimbursement rate for a Chest X-Ray is $120.00.
    1.3 CT Scan - Abdomen (procedure code 74150): The maximum allowed reimbursement rate for a CT Scan is $350.00.
    
    SECTION 2: DISALLOWED CHARGES
    2.1 Administrative Fees: Separate administrative fees, registration fees, processing charges, or intake fees are strictly disallowed.
    2.2 Facility Admin Fee: Any charges billed as 'Facility Admin Fee' or 'Facility Processing Fee' are non-reimbursable.
    2.3 Reading Fees: Reading fees are included in the primary procedure rate and cannot be billed separately.
    2.4 Late Fees: Late payment fees or interest charges are not allowed.
    
    SECTION 3: IDENTITY & COMPLIANCE
    3.1 Patient Identity: All invoices must strictly match the patient name and date of birth authorized in the referral document.
    3.2 Performing Facility: Services must be rendered at the specific hospital or clinic branch approved in the referral.
    """
    
    rect = fitz.Rect(50, 50, 550, 750)
    page.insert_textbox(rect, contract_text.strip(), fontsize=11, fontname="helv")
    doc.save(filepath)
    doc.close()

# Automatically verify/generate static contract PDF
if not os.path.exists(STATIC_CONTRACT_PATH):
    try:
        generate_default_contract(STATIC_CONTRACT_PATH)
    except Exception as e:
        st.error(f"Error creating default contract placeholder: {str(e)}")

# Initialize SQLite database
if "db_initialized" not in st.session_state:
    database.init_db()
    st.session_state["db_initialized"] = True

# Session state initialization
if "referral_data" not in st.session_state:
    st.session_state["referral_data"] = None
if "invoice_data" not in st.session_state:
    st.session_state["invoice_data"] = None
if "raw_referral_text" not in st.session_state:
    st.session_state["raw_referral_text"] = ""
if "raw_invoice_text" not in st.session_state:
    st.session_state["raw_invoice_text"] = ""
if "audit_results" not in st.session_state:
    st.session_state["audit_results"] = None
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = None
if "ticket_id" not in st.session_state:
    st.session_state["ticket_id"] = None
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "contract_indexed" not in st.session_state:
    st.session_state["contract_indexed"] = False

# Retrieve API Key from .env
api_key = os.getenv("GEMINI_API_KEY", "")

# Verify key
if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in the backend env environment. Please define it in your .env file.")

# Instantiating engines with backend API key
extractor = AuditExtractor(api_key=api_key) if api_key else None
rag_engine = ContractRAGEngine(persist_dir="./chroma_db")
audit_engine = AuditChecklistEngine(rag_engine=rag_engine, api_key=api_key) if api_key else None

# Automatically index the static contract PDF on startup
if api_key and not st.session_state["contract_indexed"]:
    if os.path.exists(STATIC_CONTRACT_PATH):
        try:
            with open(STATIC_CONTRACT_PATH, "rb") as f:
                contract_bytes = f.read()
            contract_text = file_reader.read_pdf(contract_bytes)
            
            # Index if ChromaDB does not have it
            num_clauses = rag_engine.index_contract(contract_text)
            st.session_state["contract_indexed"] = True
            logger_indexed = True
        except Exception as e:
            st.error(f"Failed to auto-index backend contract: {str(e)}")
    else:
        st.error("Static contract file ('contract.pdf') was not found in the root directory.")

# Sidebar config
st.sidebar.image("https://img.icons8.com/external-flatart-icons-lineal-color-flatart-icons/128/external-medical-insurance-medical-health-flatart-icons-lineal-color-flatart-icons.png", width=80)
st.sidebar.title("Auditing Engine")
st.sidebar.markdown("**Environment**: Production Backend")

if st.session_state["contract_indexed"]:
    st.sidebar.success("📚 **Legal Contract**: Loaded & Indexed (Static)")
else:
    st.sidebar.warning("📚 **Legal Contract**: Not Indexed")

st.sidebar.divider()
st.sidebar.subheader("Document Ingestion")

# File Uploaders
referral_file = st.sidebar.file_uploader("Upload Patient Referral (PDF)", type=["pdf"])
invoice_file = st.sidebar.file_uploader("Upload Final Invoice (PDF)", type=["pdf"])

# Main View
st.markdown('<div class="main-header">Healthcare Auditing Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Medical Billing Audit & Compliance Verification</div>', unsafe_allow_html=True)

# Check if we can run
can_audit = api_key and referral_file and invoice_file and st.session_state["contract_indexed"]

if not can_audit:
    # Onboarding view
    st.info("👋 Welcome! To begin the audit process, please upload the patient documents in the sidebar:")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### 🩺 Step 1\nUpload the **Patient Referral PDF** in the sidebar.")
    with cols[1]:
        st.markdown("### 💵 Step 2\nUpload the **Final Medical Invoice PDF**, then click 'Run Audit'.")
else:
    # Display Action Button
    if st.button("🚀 Run Comprehensive Billing Audit", use_container_width=True):
        with st.spinner("⏳ Extracting data and cross-referencing contract policy..."):
            try:
                # 1. Read files
                ref_bytes = referral_file.read()
                inv_bytes = invoice_file.read()
                
                raw_ref = file_reader.read_pdf(ref_bytes)
                raw_inv = file_reader.read_pdf(inv_bytes)
                
                st.session_state["raw_referral_text"] = raw_ref
                st.session_state["raw_invoice_text"] = raw_inv
                
                # 2. Extract structured JSON
                ref_data = extractor.extract_referral(raw_ref)
                inv_data = extractor.extract_invoice(raw_inv)
                
                st.session_state["referral_data"] = ref_data
                st.session_state["invoice_data"] = inv_data
                
                # 3. Database Sync
                patient_name = ref_data.get("patient_name")
                dob = ref_data.get("dob")
                approved_test = ref_data.get("approved_test")
                
                # Save patient, open ticket
                patient_id = database.get_or_create_patient(patient_name, dob)
                ticket_id = database.create_ticket(patient_id, referral_test=approved_test)
                
                st.session_state["patient_id"] = patient_id
                st.session_state["ticket_id"] = ticket_id
                
                # 4. Perform Audit Checklist
                results = audit_engine.run_audit(
                    ref_data, 
                    inv_data, 
                    raw_ref, 
                    raw_inv
                )
                
                st.session_state["audit_results"] = results
                
                # Calculate overall audit ticket status
                any_fails = any(r["status"] == "Fail" for r in results)
                final_status = "Failed" if any_fails else "Passed"
                
                # Update SQLite ticket
                invoice_total = inv_data.get("total_amount", 0.0)
                database.update_ticket(ticket_id, final_status, invoice_total, results)
                
                # Clear chat history on new audit
                st.session_state["chat_messages"] = []
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Audit Execution Failed: {str(e)}")

# Display audit results if they exist in session state
if st.session_state["audit_results"] is not None:
    ref_data = st.session_state["referral_data"]
    inv_data = st.session_state["invoice_data"]
    results = st.session_state["audit_results"]
    
    # Side by side comparison dashboard
    st.subheader("📋 Document Extractions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Referral Extracted Data")
        st.write(f"👤 **Patient Name**: {ref_data.get('patient_name')}")
        st.write(f"📅 **Date of Birth**: {ref_data.get('dob')}")
        st.write(f"🔬 **Approved Test**: `{ref_data.get('approved_test')}`")
        st.write(f"🏢 **Authorized Facility**: `{ref_data.get('facility')}`")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Invoice Extracted Data")
        st.write(f"👤 **Patient Name**: {inv_data.get('patient_name')}")
        st.write(f"💵 **Invoice Stated Total**: `${inv_data.get('total_amount'):,.2f}`")
        st.markdown("**Billed Line Items:**")
        for item in inv_data.get("billed_services", []):
            st.write(f"- {item.get('item')}: `${item.get('cost'):,.2f}`")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.divider()
    
    # 6-Point Audit Checklist
    st.subheader("⚖️ 6-Point Audit Checklist")
    
    # Overall summary status
    failures = [r["check_name"] for r in results if r["status"] == "Fail"]
    if len(failures) == 0:
        st.success("🎉 **AUDIT STATUS: PASSED** — All billing checks matched authorized guidelines and contract pricing caps.")
    else:
        st.error(f"❌ **AUDIT STATUS: FAILED** — The invoice failed {len(failures)} verification checks: {', '.join(failures)}.")
        
    # Render each check
    for check in results:
        status_pill = (
            f'<span class="status-pill-pass">🟢 PASS</span>' 
            if check["status"] == "Pass" 
            else f'<span class="status-pill-fail">🔴 FAIL</span>'
        )
        card_class = "check-card-pass" if check["status"] == "Pass" else "check-card-fail"
        
        st.markdown(
            f'<div class="{card_class}">'
            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<div class="check-title">{check["check_name"]}</div>'
            f'<div>{status_pill}</div>'
            f'</div>'
            f'<p style="margin-top: 0.5rem; margin-bottom: 0.5rem;"><strong>Reason:</strong> {check["reason"]}</p>'
            f'<p style="font-size: 0.9rem; color: #444; margin-bottom: 0px;">📜 <strong>Contract Citation:</strong> <em>{check["contract_clause_citation"]}</em></p>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    st.divider()
    
    # Relational Database View: Patient History
    st.subheader("📜 Patient Audit Ticket History")
    history = database.get_patient_history(st.session_state["patient_id"])
    if len(history) > 1:
        st.info(f"Historical audit logs found for patient {ref_data.get('patient_name')}:")
        for hist in history:
            ticket_status_color = "green" if hist["status"] == "Passed" else "red"
            st.markdown(
                f"- **Ticket #{hist['ticket_id']}** ({hist['created_at'][:10]}) | "
                f"Test: `{hist['referral_test']}` | "
                f"Total Billed: `${hist['invoice_total']:,.2f}` | "
                f"Status: :**{ticket_status_color}[{hist['status']}]**"
            )
    else:
        st.write("This is the first recorded audit ticket for this patient.")
        
    st.divider()
    
    # Interactive Chat Interface
    st.subheader("💬 Auditor Assistant Chatbot")
    st.write("Ask the assistant specific questions about this audit (e.g. why did a fee fail, or details of the contract pricing caps).")
    
    # Display previous messages
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Input box
    if user_query := st.chat_input("Enter your question here..."):
        # Add user query to chat history
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.spinner("Thinking..."):
            try:
                # Retrieve context from contract (RAG)
                relevant_clauses = rag_engine.search_contract(user_query, n_results=3)
                rag_context = "\n\n".join([f"Contract Clause:\n{rc['text']}" for rc in relevant_clauses])
                
                # Build context
                chatbot_context = (
                    "You are an expert Medical Billing Audit Assistant. Your job is to help the auditor "
                    "understand the audit results, patient details, and billing contract policy.\n\n"
                    "=== PATIENT REFERRAL DATA ===\n"
                    f"{json.dumps(ref_data, indent=2)}\n\n"
                    "=== MEDICAL INVOICE DATA ===\n"
                    f"{json.dumps(inv_data, indent=2)}\n\n"
                    "=== 6-POINT AUDIT RESULT DETAILS ===\n"
                    f"{json.dumps(results, indent=2)}\n\n"
                    "=== RELEVANT LEGAL CONTRACT POLICY ===\n"
                    f"{rag_context}\n\n"
                    "Answer the user's question accurately using only the facts presented above. "
                    "If the answer is not contained in the context, politely state that you do not "
                    "have that information."
                )
                
                # Query Gemini
                model = genai.GenerativeModel("gemini-2.5-flash")
                chat_response = generate_content_with_retry(
                    model,
                    [
                        chatbot_context,
                        # Provide recent history for context
                        *[f"{m['role'].upper()}: {m['content']}" for m in st.session_state["chat_messages"][-5:-1]],
                        f"USER: {user_query}"
                    ]
                )
                
                assistant_response = chat_response.text.strip()
                
                # Add assistant response to history
                st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    st.write(assistant_response)
                    
            except Exception as e:
                st.error(f"Chat failed: {str(e)}")
