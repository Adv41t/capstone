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
    Executes the 6-Point Automated Audit Logic by cross-referencing
    the Referral JSON, Invoice JSON, raw texts, and querying ChromaDB (RAG).
    Includes automatic retries for rate limits (quota limits).
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
        raw_invoice_text: str
    ) -> List[Dict[str, Any]]:
        """
        Runs all 6 audit checks and returns the list of results.
        
        Returns:
            List of dicts: [{check_name, status ("Pass" or "Fail"), reason, contract_clause_citation}]
        """
        results = []

        # 1. Identity Match
        results.append(self._check_identity(referral_data, invoice_data))

        # 2. Test Match
        results.append(self._check_test_match(referral_data, invoice_data))

        # 3. Facility Match
        results.append(self._check_facility_match(referral_data, raw_invoice_text))

        # 4. Allowed Fees
        results.append(self._check_allowed_fees(invoice_data))

        # 5. Pricing Cap
        results.append(self._check_pricing_cap(referral_data, invoice_data))

        # 6. Total Calculation Validation
        results.append(self._check_total_calculation(invoice_data))

        return results

    def _check_identity(self, referral_data: Dict[str, Any], invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Identity Match: Does the name on the referral match the name on the bill?
        """
        ref_name = referral_data.get("patient_name", "").strip()
        inv_name = invoice_data.get("patient_name", "").strip()

        if not ref_name or not inv_name:
            return {
                "check_name": "Identity Match",
                "status": "Fail",
                "reason": f"Missing patient name. Referral: '{ref_name}', Invoice: '{inv_name}'",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

        # Exact check
        if ref_name.lower() == inv_name.lower():
            return {
                "check_name": "Identity Match",
                "status": "Pass",
                "reason": f"Patient names match exactly: '{ref_name}'",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

        # Fuzzy check via Gemini for variations (e.g. John A. Doe vs John Doe)
        model = self._get_model()
        prompt = (
            "Evaluate if the following two names represent the same patient. "
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
            reason = res.get("reason", "Name comparison complete.")
            return {
                "check_name": "Identity Match",
                "status": status,
                "reason": reason,
                "contract_clause_citation": "N/A (Standard Verification)"
            }
        except Exception as e:
            logger.error(f"Fuzzy name match failed: {str(e)}")
            # Fallback to simple sub-string comparison
            if ref_name.lower() in inv_name.lower() or inv_name.lower() in ref_name.lower():
                return {
                    "check_name": "Identity Match",
                    "status": "Pass",
                    "reason": f"Patient names are highly similar: Referral: '{ref_name}', Invoice: '{inv_name}' (Substring Match)",
                    "contract_clause_citation": "N/A (Standard Verification)"
                }
            return {
                "check_name": "Identity Match",
                "status": "Fail",
                "reason": f"Patient names do not match. Referral: '{ref_name}', Invoice: '{inv_name}'",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

    def _check_test_match(self, referral_data: Dict[str, Any], invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        2. Test Match: Is the billed test the exact one authorized in the referral?
        """
        approved_test = referral_data.get("approved_test", "").strip()
        billed_services = invoice_data.get("billed_services", [])

        if not approved_test:
            return {
                "check_name": "Test Match",
                "status": "Fail",
                "reason": "Approved test is missing from the referral data.",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

        if not billed_services:
            return {
                "check_name": "Test Match",
                "status": "Fail",
                "reason": "No billed services found on the invoice.",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

        # Check semantic overlap using Gemini
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
                "check_name": "Test Match",
                "status": status,
                "reason": reason,
                "contract_clause_citation": "N/A (Standard Verification)"
            }
        except Exception as e:
            logger.error(f"Semantic test match failed: {str(e)}")
            # Simple substring fallback
            for service in billed_services:
                item = service.get("item", "")
                if approved_test.lower() in item.lower() or item.lower() in approved_test.lower():
                    return {
                        "check_name": "Test Match",
                        "status": "Pass",
                        "reason": f"Matched authorized test '{approved_test}' with billed service '{item}' (Fallback Match)",
                        "contract_clause_citation": "N/A (Standard Verification)"
                    }
            return {
                "check_name": "Test Match",
                "status": "Fail",
                "reason": f"Authorized test '{approved_test}' could not be matched with any billed service.",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

    def _check_facility_match(self, referral_data: Dict[str, Any], raw_invoice_text: str) -> Dict[str, Any]:
        """
        3. Facility Match: Was the test performed at the facility authorized in the referral?
        """
        approved_facility = referral_data.get("facility", "").strip()

        if not approved_facility:
            return {
                "check_name": "Facility Match",
                "status": "Fail",
                "reason": "Authorized facility is missing from the referral data.",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

        # Query Gemini to extract performing facility from the raw invoice text and compare
        model = self._get_model()
        prompt = (
            f"Authorized Facility in Referral: '{approved_facility}'\n"
            f"Raw Invoice Text:\n{raw_invoice_text}\n\n"
            "Find the facility name where the invoice services were performed or where the invoice was issued from. "
            "Compare it with the authorized facility. Are they the same facility (allow minor abbreviations or hospital branch names)?\n"
            "Return a JSON response matching this schema:\n"
            "{\n"
            '  "performing_facility": "extracted performing facility from invoice",\n'
            '  "match": true/false,\n'
            '  "reason": "explanation of comparison"\n'
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
            reason = res.get("reason", f"Performing Facility: {res.get('performing_facility')}")
            return {
                "check_name": "Facility Match",
                "status": status,
                "reason": reason,
                "contract_clause_citation": "N/A (Standard Verification)"
            }
        except Exception as e:
            logger.error(f"Facility check failed: {str(e)}")
            return {
                "check_name": "Facility Match",
                "status": "Fail",
                "reason": f"Could not verify if invoice was issued from '{approved_facility}' due to verification error: {str(e)}",
                "contract_clause_citation": "N/A (Standard Verification)"
            }

    def _check_allowed_fees(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        4. Allowed Fees: Are there any disallowed line items on the bill?
        Queries RAG to see if each fee item is explicitly disallowed.
        """
        billed_services = invoice_data.get("billed_services", [])
        if not billed_services:
            return {
                "check_name": "Allowed Fees",
                "status": "Fail",
                "reason": "No billed services found on the invoice.",
                "contract_clause_citation": "N/A"
            }

        disallowed_items = []
        citations = []
        model = self._get_model()

        for service in billed_services:
            item_name = service.get("item", "")
            
            # Query RAG Engine for this item
            query = f"Are {item_name} allowed? List disallowed fees or policy on administrative charges."
            clauses = self.rag_engine.search_contract(query, n_results=2)
            
            context = "\n\n".join([f"Clause:\n{c['text']}" for c in clauses])
            
            prompt = (
                f"We are auditing a medical bill with line item: '{item_name}'\n"
                "Verify if this type of fee is disallowed or forbidden according to the contract clauses below.\n"
                "For example, contracts often disallow separate administrative fees, facility fees, or check-in charges.\n\n"
                f"Relevant Contract Clauses:\n{context}\n\n"
                "Return a JSON response matching this schema:\n"
                "{\n"
                '  "is_allowed": true/false,\n'
                '  "reason": "explanation citing details from the contract clauses",\n'
                '  "citation": "direct quote or clause identifier from the contract context"\n'
                "}"
            )
            try:
                response = generate_content_with_retry(
                    model,
                    prompt,
                    generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
                )
                res = json.loads(response.text)
                if not res.get("is_allowed"):
                    disallowed_items.append(f"'{item_name}' ({res.get('reason')})")
                    citations.append(res.get("citation", ""))
            except Exception as e:
                logger.error(f"Disallowed fee check failed for '{item_name}': {str(e)}")
                # Reraise so checklist can fail gracefully rather than masking actual exceptions
                raise e

        if disallowed_items:
            return {
                "check_name": "Allowed Fees",
                "status": "Fail",
                "reason": f"Disallowed items found: {'; '.join(disallowed_items)}",
                "contract_clause_citation": " | ".join(filter(None, citations)) or "Refer to contract policy on disallowed administrative fees."
            }
        
        return {
            "check_name": "Allowed Fees",
            "status": "Pass",
            "reason": "All billed line items appear to be contractually permitted.",
            "contract_clause_citation": "Refer to contract terms regarding approved pricing and medical billing guidelines."
        }

    def _check_pricing_cap(self, referral_data: Dict[str, Any], invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        5. Pricing Cap: Are the billed amounts within the allowed maximums?
        Queries RAG: "What is the maximum allowed rate for [Test Name]?"
        """
        approved_test = referral_data.get("approved_test", "").strip()
        billed_services = invoice_data.get("billed_services", [])

        if not approved_test or not billed_services:
            return {
                "check_name": "Pricing Cap",
                "status": "Fail",
                "reason": "Cannot run pricing cap check due to missing test or invoice data.",
                "contract_clause_citation": "N/A"
            }

        # 1. Search contract for rates related to this test
        query = f"What is the maximum allowed rate, fee schedule, or reimbursement cost for {approved_test}?"
        clauses = self.rag_engine.search_contract(query, n_results=3)
        context = "\n\n".join([f"Clause:\n{c['text']}" for c in clauses])

        # 2. Get the specific billed amount for the main test
        model = self._get_model()
        billed_items_str = json.dumps(billed_services)
        
        prompt = (
            f"Approved Test: '{approved_test}'\n"
            f"Billed Items on Invoice:\n{billed_items_str}\n\n"
            "Based on the following contract clauses, identify the pricing cap (maximum allowed rate) for the approved test. "
            "Then, identify the matching billed item(s) on the invoice and determine if the billed cost exceeds that cap.\n\n"
            f"Relevant Contract Clauses:\n{context}\n\n"
            "Return a JSON response matching this schema:\n"
            "{\n"
            '  "pricing_cap_found": true/false,\n'
            '  "pricing_cap_amount": 0.0, -- maximum allowed rate as a number\n'
            '  "exceeds_cap": true/false,\n'
            '  "reason": "explanation of contract rate matching and cost comparison",\n'
            '  "citation": "exact text from the contract clause specifying the rate"\n'
            "}"
        )
        try:
            response = generate_content_with_retry(
                model,
                prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            res = json.loads(response.text)
            
            if not res.get("pricing_cap_found"):
                return {
                    "check_name": "Pricing Cap",
                    "status": "Pass",
                    "reason": "No pricing cap or fee schedule was found in the contract for this specific test.",
                    "contract_clause_citation": "N/A"
                }
                
            status = "Fail" if res.get("exceeds_cap") else "Pass"
            reason = res.get("reason", "Pricing cap verification complete.")
            citation = res.get("citation", "Refer to Contract Fee Schedule.")
            
            return {
                "check_name": "Pricing Cap",
                "status": status,
                "reason": reason,
                "contract_clause_citation": citation
            }
        except Exception as e:
            logger.error(f"Pricing cap check failed: {str(e)}")
            raise e

    def _check_total_calculation(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        6. Total Calculation Validation: Does the sum of the line items equal the total amount stated?
        """
        billed_services = invoice_data.get("billed_services", [])
        stated_total = invoice_data.get("total_amount")

        if stated_total is None:
            return {
                "check_name": "Total Calculation Validation",
                "status": "Fail",
                "reason": "Invoice total amount is missing or could not be extracted.",
                "contract_clause_citation": "N/A (Standard Arithmetic Check)"
            }

        calculated_sum = sum(s.get("cost", 0.0) for s in billed_services)
        
        if abs(calculated_sum - stated_total) <= 0.01:
            return {
                "check_name": "Total Calculation Validation",
                "status": "Pass",
                "reason": f"Calculated sum of line items (${calculated_sum:,.2f}) matches the stated total (${stated_total:,.2f}) exactly.",
                "contract_clause_citation": "N/A (Standard Arithmetic Check)"
            }
        else:
            return {
                "check_name": "Total Calculation Validation",
                "status": "Fail",
                "reason": f"Sum of line items is ${calculated_sum:,.2f}, but the invoice states a total of ${stated_total:,.2f}. Difference is ${abs(calculated_sum - stated_total):,.2f}.",
                "contract_clause_citation": "N/A (Standard Arithmetic Check)"
            }
