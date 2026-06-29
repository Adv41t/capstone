import os
import json
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import google.api_core.exceptions
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

# Setup logging
logging.basicConfig(level=logging.INFO)
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
    logger.info("Attempting Gemini API generation...")
    return model.generate_content(*args, **kwargs)


# Pydantic Schemas for Strict Gemini Output Enforcements
class ReferralData(BaseModel):
    patient_name: str = Field(description="Full name of the patient (First Last).")
    dob: str = Field(description="Date of birth of the patient (e.g. YYYY-MM-DD or MM/DD/YYYY).")
    approved_test: str = Field(description="The specific medical test, procedure, or scan authorized in the referral.")
    facility: str = Field(description="The specific clinic, hospital, or facility authorized to perform the test.")

class BilledService(BaseModel):
    item: str = Field(description="Description of the billed service, procedure, fee, adjustment, discount, or credit.")
    cost: float = Field(description="The cost/price charged (use negative value for adjustments, discounts, or insurance credits that reduce the total).")

class InvoiceData(BaseModel):
    patient_name: str = Field(description="Full name of the patient (First Last) listed on the invoice.")
    billed_services: List[BilledService] = Field(description="List of all services/items billed on the invoice.")
    total_amount: float = Field(description="The final total amount billed stated on the invoice.")


class AuditExtractor:
    """
    Handles structured data extraction from raw documents (referrals & invoices)
    using the Gemini API and strict JSON schemas with rate limit retries.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the extractor with the Gemini API key.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def _get_model(self, model_name: str = "gemini-2.5-flash") -> genai.GenerativeModel:
        """Helper to get the configured GenerativeModel."""
        if not self.api_key:
            raise ValueError("Gemini API key is not configured. Please set GEMINI_API_KEY environment variable.")
        return genai.GenerativeModel(model_name)

    def extract_referral(self, text: str) -> Dict[str, Any]:
        """
        Extracts patient name, DOB, approved test, and facility from referral text.
        Guarantees response conforms to ReferralData schema.
        """
        if not text.strip():
            raise ValueError("Referral text is empty. Cannot extract data.")

        model = self._get_model()
        prompt = (
            "Analyze the following medical referral document and extract the patient's information, "
            "the approved test or procedure, and the authorized facility.\n\n"
            f"Referral Document Content:\n{text}"
        )

        try:
            response = generate_content_with_retry(
                model,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ReferralData,
                    temperature=0.1
                )
            )
            
            # Ensure the response is valid JSON matching the schema
            data = json.loads(response.text)
            logger.info("Successfully extracted referral data.")
            return data
            
        except json.JSONDecodeError as jde:
            logger.error(f"Failed to parse Gemini output as JSON: {response.text}")
            raise RuntimeError("Gemini did not return valid JSON for referral data.") from jde
        except Exception as e:
            logger.error(f"Error extracting referral data: {str(e)}")
            raise RuntimeError(f"Referral extraction failed: {str(e)}") from e

    def extract_invoice(self, text: str) -> Dict[str, Any]:
        """
        Extracts patient name, billed services, and total amount from invoice text.
        Guarantees response conforms to InvoiceData schema.
        """
        if not text.strip():
            raise ValueError("Invoice text is empty. Cannot extract data.")

        model = self._get_model()
        prompt = (
            "Analyze the following medical invoice/bill and extract the patient name, "
            "all billed line items (each with an item description and cost—be sure to extract "
            "adjustments, insurance write-offs, discounts, or credits with negative values if they "
            "reduce the net total), and the final net total amount billed/due stated on the bill.\n\n"
            f"Invoice Document Content:\n{text}"
        )

        try:
            response = generate_content_with_retry(
                model,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=InvoiceData,
                    temperature=0.1
                )
            )
            
            data = json.loads(response.text)
            logger.info("Successfully extracted invoice data.")
            return data
            
        except json.JSONDecodeError as jde:
            logger.error(f"Failed to parse Gemini output as JSON: {response.text}")
            raise RuntimeError("Gemini did not return valid JSON for invoice data.") from jde
        except Exception as e:
            logger.error(f"Error extracting invoice data: {str(e)}")
            raise RuntimeError(f"Invoice extraction failed: {str(e)}") from e
