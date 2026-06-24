import os
from typing import Union
import fitz  # PyMuPDF

def read_pdf(file_input: Union[str, bytes]) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF.

    Args:
        file_input: Path to the PDF file or the raw bytes of the file.

    Returns:
        The extracted text content from all pages of the PDF.
    
    Raises:
        RuntimeError: If PDF parsing fails.
    """
    try:
        if isinstance(file_input, bytes):
            doc = fitz.open(stream=file_input, filetype="pdf")
        else:
            if not os.path.exists(file_input):
                raise FileNotFoundError(f"PDF file not found at: {file_input}")
            doc = fitz.open(file_input)
            
        text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text:
                text.append(page_text)
                
        doc.close()
        return "\n".join(text)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {str(e)}") from e

