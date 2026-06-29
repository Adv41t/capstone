import os
import json
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import google.api_core.exceptions
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception
from rag_engine import ContractRAGEngine

logger = logging.getLogger(__name__)

def is_rate_limit_error(exception):
    """Determines if the exception is due to API rate limits or quota exhaustion."""
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
    """Calls model.generate_content with exponential backoff on rate limit errors."""
    logger.info("Attempting Gemini API checklist verification...")
    return model.generate_content(*args, **kwargs)


class AuditChecklistEngine:
    """
    Executes the restructured 6-Point Automated Workflow & Audit Logic by
    cross-referencing Referral data, Invoice data, raw texts, and legal contract RAG.
    """
    def __init__(self, rag_engine: ContractRAGEngine, api_key: Optional[str] = None):
        """
        Initializes the audit engine with the RAG engine and Gemini API credentials.
        """
        self.rag_engine = rag_engine
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def _get_model(self, model_name: str = "gemini-2.5-flash") -> genai.GenerativeModel:
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")
        return genai.GenerativeModel(model_name)

    def run_audit(
        self,
        referral_data: Dict[str, Any],
        invoice_data: Dict[str, Any],
        raw_referral_text: str,
        raw_invoice_text: str,
        pymupdf_success: bool = True,
        pydantic_success: bool = True,
        extraction_error: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Runs the 6 workflow and audit checks and returns the list of results.
        
        Returns:
            List of dicts: [{check_name, status ("Pass" or "Fail"), reason, contract_clause_citation}]
        """
        results = []

        # 1. Files fetched (Workflow check)
        files_fetched_res = self._check_files_fetched(pymupdf_success, pydantic_success, extraction_error)
        results.append(files_fetched_res)

        # If files were not successfully fetched/parsed, the remaining checks must fail due to missing data.
        if files_fetched_res["status"] == "Fail":
            remaining_checks = [
                "Patient matched",
                "Insurance applied",
                "Test Matched",
                "Allowed fees and Pricing cap",
                "Total calculation validation"
            ]
            for check in remaining_checks:
                results.append({
                    "check_name": check,
                    "status": "Fail",
                    "reason": "This check could not be run because document text or data extraction failed.",
                    "contract_clause_citation": "N/A"
                })
            return results

        # 2. Patient matched (Gemini comparison)
        results.append(self._check_patient_matched(referral_data, invoice_data))

        # 3. Insurance applied? (PyMuPDF/Gemini verification)
        results.append(self._check_insurance_applied(raw_invoice_text))

        # 4. Test Matched (Gemini comparison)
        results.append(self._check_test_matched(referral_data, invoice_data))

        # 5. Allowed fees and pricing cap (RAG + Gemini comparison)
        results.append(self._check_allowed_fees_and_pricing_cap(referral_data, invoice_data))

        # 6. Total calculation validation (Basic mathematics)
        results.append(self._check_total_calculation(invoice_data))

        return results

    def _check_files_fetched(self, pymupdf_success: bool, pydantic_success: bool, extraction_error: str) -> Dict[str, Any]:
        """
        1. Files fetched: Verifies if the texts have been extracted by PyMuPDF and parsed successfully via Pydantic.
        """
        if pymupdf_success and pydantic_success:
            return {
                "check_name": "Files fetched",
                "status": "Pass",
                "reason": "Documents successfully loaded via PyMuPDF and parsed into JSON matching our schemas.",
                "contract_clause_citation": "N/A (Workflow Checklist)"
            }
        else:
            errors = []
            if not pymupdf_success:
                errors.append("PyMuPDF failed to extract text from the PDF files.")
            if not pydantic_success:
                errors.append("Pydantic failed to map the extracted text into structured JSON.")
            if extraction_error:
                errors.append(extraction_error)
            return {
                "check_name": "Files fetched",
                "status": "Fail",
                "reason": f"Workflow failed at the ingestion stage: {' '.join(errors)}",
                "contract_clause_citation": "N/A (Workflow Checklist)"
            }

    def _check_patient_matched(self, referral_data: Dict[str, Any], invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        2. Patient matched: Does the patient name on the referral match the patient name on the bill?
        Uses Gemini comparison for fuzzy matching.
        """
        ref_name = referral_data.get("patient_name", "").strip()
        inv_name = invoice_data.get("patient_name", "").strip()

        if not ref_name or not inv_name:
            return {
                "check_name": "Patient matched",
                "status": "Fail",
                "reason": f"Missing patient name in extracted data. Referral: '{ref_name}', Invoice: '{inv_name}'",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

        # Fuzzy check via Gemini for variations (e.g. John A. Doe vs John Doe)
        model = self._get_model()
        prompt = (
            "Evaluate if the following two patient names represent the same person. "
            "Address minor variations like abbreviations, middle names, or minor spelling errors.\n"
            f"Referral Patient Name: {ref_name}\n"
            f"Invoice Patient Name: {inv_name}\n\n"
            "Return a JSON response matching this schema:\n"
            "{\n"
            '  "match": true/false,\n'
            '  "reason": "explanation of the evaluation"\n'
            "}"
        )
        try:
            response = generate_content_with_retry(
                model,
                prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            res = json.loads(response.text)
            status = "Pass" if res.get("match") else "Fail"
            reason = res.get("reason", "Patient comparison complete.")
            return {
                "check_name": "Patient matched",
                "status": status,
                "reason": reason,
                "contract_clause_citation": "N/A (Standard Verification)"
            }
        except Exception as e:
            logger.error(f"Patient fuzzy match failed: {str(e)}")
            # Fallback to simple substring comparison
            if ref_name.lower() in inv_name.lower() or inv_name.lower() in ref_name.lower():
                return {
                    "check_name": "Patient matched",
                    "status": "Pass",
                    "reason": f"Patient names are highly similar: Referral: '{ref_name}', Invoice: '{inv_name}' (Fallback Match)",
                    "contract_clause_citation": "N/A (Standard Verification)"
                }
            return {
                "check_name": "Patient matched",
                "status": "Fail",
                "reason": f"Patient names do not match. Referral: '{ref_name}', Invoice: '{inv_name}'",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

    def _check_insurance_applied(self, raw_invoice_text: str) -> Dict[str, Any]:
        """
        3. Insurance applied?: Check if the insurance applied box/field is ticked/YES or NO.
        Uses Gemini to analyze raw text extracted by PyMuPDF.
        """
        model = self._get_model()
        prompt = (
            "Analyze the raw invoice text below and determine if insurance coverage was applied. "
            "Identify if an insurance applied checkbox, status field, or indicator is set to YES (ticked) or NO (unticked).\n"
            "Examples: 'Insurance Coverage Applied? YES ✓', 'Insurance Coverage Applied? YES', 'AWAITING INSURANCE', "
            "or transaction ledgers listing insurance adjustments or payments indicate YES. If insurance is not listed or explicitly NO, choose false.\n\n"
            f"Raw Invoice Text:\n{raw_invoice_text}\n\n"
            "Return a JSON response matching this schema:\n"
            "{\n"
            '  "insurance_applied": true/false,\n'
            '  "reason": "explanation of how you determined if insurance was applied (Yes or No), quoting relevant text from the invoice"\n'
            "}"
        )
        try:
            response = generate_content_with_retry(
                model,
                prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            res = json.loads(response.text)
            status = "Pass" if res.get("insurance_applied") else "Fail"
            reason = res.get("reason", "Insurance status checked.")
            return {
                "check_name": "Insurance applied?",
                "status": status,
                "reason": reason,
                "contract_clause_citation": "N/A (Standard Verification)"
            }
        except Exception as e:
            logger.error(f"Insurance check failed: {str(e)}")
            lower_text = raw_invoice_text.lower()
            if "insurance coverage applied?* yes" in lower_text or "insurance primary" in lower_text:
                return {
                    "check_name": "Insurance applied?",
                    "status": "Pass",
                    "reason": "Found indication of insurance coverage applied in raw text (Fallback).",
                    "contract_clause_citation": "N/A (Standard Verification)"
                }
            return {
                "check_name": "Insurance applied?",
                "status": "Fail",
                "reason": f"Could not determine if insurance coverage was applied. Details: {str(e)}",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

    def _check_test_matched(self, referral_data: Dict[str, Any], invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        4. Test Matched: Is the billed test/procedure the exact one authorized in the referral?
        Uses Gemini comparison for semantic validation.
        """
        approved_test = referral_data.get("approved_test", "").strip()
        billed_services = invoice_data.get("billed_services", [])

        if not approved_test:
            return {
                "check_name": "Test Matched",
                "status": "Fail",
                "reason": "Approved test is missing from the referral data.",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

        if not billed_services:
            return {
                "check_name": "Test Matched",
                "status": "Fail",
                "reason": "No billed services found on the invoice.",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

        model = self._get_model()
        billed_items_str = ", ".join([f"'{s.get('item')}'" for s in billed_services])
        prompt = (
            f"Authorized Test in Referral: '{approved_test}'\n"
            f"Billed Items in Invoice: [{billed_items_str}]\n\n"
            "Evaluate if the authorized test is present among the billed invoice items. "
            "Note that the billed items may use slightly different wording, abbreviations, or codes "
            "(e.g., 'Brain MRI' matches 'MRI - Brain without contrast').\n"
            "Return a JSON response matching this schema:\n"
            "{\n"
            '  "match": true/false,\n'
            '  "reason": "explanation of mapping or failure",\n'
            '  "matched_billed_item": "the matching item name from the billed list, or empty if none"\n'
            "}"
        )
        try:
            response = generate_content_with_retry(
                model,
                prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            res = json.loads(response.text)
            status = "Pass" if res.get("match") else "Fail"
            reason = res.get("reason", "Test verification complete.")
            return {
                "check_name": "Test Matched",
                "status": status,
                "reason": reason,
                "contract_clause_citation": "N/A (Standard Verification)"
            }
        except Exception as e:
            logger.error(f"Semantic test match failed: {str(e)}")
            # Fallback substring matching
            for service in billed_services:
                item = service.get("item", "")
                if approved_test.lower() in item.lower() or item.lower() in approved_test.lower():
                    return {
                        "check_name": "Test Matched",
                        "status": "Pass",
                        "reason": f"Matched authorized test '{approved_test}' with billed service '{item}' (Fallback Match)",
                        "contract_clause_citation": "N/A (Standard Verification)"
                    }
            return {
                "check_name": "Test Matched",
                "status": "Fail",
                "reason": f"Authorized test '{approved_test}' could not be matched with any billed service.",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

    def _check_allowed_fees_and_pricing_cap(
        self, referral_data: Dict[str, Any], invoice_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        5. Allowed fees and pricing cap: Are all billed items contractually allowed, and are cost amounts within the pricing cap?
        Uses RAG to fetch relevant contract clauses, and Gemini to perform combined evaluation.
        """
        approved_test = referral_data.get("approved_test", "").strip()
        billed_services = invoice_data.get("billed_services", [])

        if not approved_test or not billed_services:
            return {
                "check_name": "Allowed fees and pricing cap",
                "status": "Fail",
                "reason": "Cannot run allowed fees and pricing cap check due to missing test or invoice data.",
                "contract_clause_citation": "N/A"
            }

        # RAG search for disallowed fees policy (e.g. administrative fees, facility admin fees, processing fees)
        disallowed_query = "Are administrative fees, processing charges, facility admin fees, or reading fees disallowed?"
        disallowed_clauses = self.rag_engine.search_contract(disallowed_query, n_results=2)

        # RAG search for pricing caps related to this approved test
        pricing_query = f"What is the maximum allowed rate, fee schedule, or reimbursement cost for {approved_test}?"
        pricing_clauses = self.rag_engine.search_contract(pricing_query, n_results=3)

        # Merge and deduplicate contract clauses
        all_clauses = disallowed_clauses + pricing_clauses
        seen_ids = set()
        unique_clauses = []
        for c in all_clauses:
            if c['id'] not in seen_ids:
                seen_ids.add(c['id'])
                unique_clauses.append(c)

        context = "\n\n".join([f"Clause (ID: {c['id']}):\n{c['text']}" for c in unique_clauses])

        model = self._get_model()
        billed_items_str = json.dumps(billed_services)

        prompt = (
            f"Approved Test: '{approved_test}'\n"
            f"Billed Items on Invoice:\n{billed_items_str}\n\n"
            "Based on the following contract clauses, evaluate two things:\n"
            "1. Are there any disallowed charges or fees on the invoice? (e.g. administrative charges, facility fees, reading fees, or other disallowed items specified in the contract).\n"
            "2. Is the billed amount for the approved test within the pricing cap (maximum allowed rate) specified in the contract?\n\n"
            f"Relevant Contract Clauses:\n{context}\n\n"
            "Return a JSON response matching this schema:\n"
            "{\n"
            '  "disallowed_fees_found": true/false,\n'
            '  "disallowed_fees_details": "explanation of any disallowed fees found, or empty if none",\n'
            '  "pricing_cap_exceeded": true/false,\n'
            '  "pricing_cap_details": "explanation of pricing cap comparison, including the cap amount and billed amount, or stating if no cap was found",\n'
            '  "citation": "direct quote or clause identifier from the contract clauses supporting this check"\n'
            "}"
        )

        try:
            response = generate_content_with_retry(
                model,
                prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            res = json.loads(response.text)

            disallowed = res.get("disallowed_fees_found", False)
            exceeded = res.get("pricing_cap_exceeded", False)

            status = "Fail" if (disallowed or exceeded) else "Pass"
            
            reasons = []
            if disallowed:
                reasons.append(f"Disallowed fees: {res.get('disallowed_fees_details')}")
            else:
                reasons.append("All billed fees are allowed.")

            if exceeded:
                reasons.append(f"Pricing cap: {res.get('pricing_cap_details')}")
            else:
                reasons.append(f"Pricing cap check: {res.get('pricing_cap_details', 'Billed rates within limits.')}")

            reason = " | ".join(reasons)
            citation = res.get("citation", "Refer to Contract Fee Schedule & Disallowed Fees Policy.")

            return {
                "check_name": "Allowed fees and pricing cap",
                "status": status,
                "reason": reason,
                "contract_clause_citation": citation
            }
        except Exception as e:
            logger.error(f"Allowed fees and pricing cap check failed: {str(e)}")
            raise e

    def _check_total_calculation(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        6. Total calculation validation: Does the sum of the line items equal the total amount stated?
        Uses basic mathematics.
        """
        billed_services = invoice_data.get("billed_services", [])
        stated_total = invoice_data.get("total_amount")

        if stated_total is None:
            return {
                "check_name": "Total calculation validation",
                "status": "Fail",
                "reason": "Invoice total amount is missing or could not be extracted.",
                "contract_clause_citation": "N/A (Standard Arithmetic Check)"
            }

        calculated_sum = sum(s.get("cost", 0.0) for s in billed_services)
        
        if abs(calculated_sum - stated_total) <= 0.01:
            return {
                "check_name": "Total calculation validation",
                "status": "Pass",
                "reason": f"Calculated sum of line items (${calculated_sum:,.2f}) matches the stated total (${stated_total:,.2f}) exactly.",
                "contract_clause_citation": "N/A (Standard Arithmetic Check)"
            }
        else:
            return {
                "check_name": "Total calculation validation",
                "status": "Fail",
                "reason": f"Sum of line items is ${calculated_sum:,.2f}, but the invoice states a total of ${stated_total:,.2f}. Difference is ${abs(calculated_sum - stated_total):,.2f}.",
                "contract_clause_citation": "N/A (Standard Arithmetic Check)"
            }
