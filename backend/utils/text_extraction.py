import pdfplumber
from docx import Document

def extract_text(file_path: str) -> str:
    """
    Extracts text from a PDF or DOCX file.
    """
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(docx_path: str) -> str:
    """
    Extracts text from a DOCX file.
    """
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()
