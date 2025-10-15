# utils.py
import PyPDF2
from docx import Document
import re

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def highlight_text(text, query):
    if not query:
        return text
    pattern = re.escape(query)
    highlighted = re.sub(f'({pattern})', r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return highlighted