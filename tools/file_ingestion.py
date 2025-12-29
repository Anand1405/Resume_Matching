from strands import tool
import os
import io
import docx2txt
import pdfplumber

def read_file_content(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == '.pdf':
            with pdfplumber.open(filepath) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        elif ext == '.docx':
            return docx2txt.process(filepath)
        elif ext == '.txt':
            with open(filepath, "r", encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Unsupported format: {filepath}")
            return ""
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return ""

@tool(name="read_resume_file", description="Reads the content of a resume file (PDF, DOCX, TXT) and returns the text.")
def read_resume_file(filepath: str) -> str:
    """
    Reads the content of a resume file (PDF, DOCX, TXT) and returns the text.

    Args:
        filepath: Path to the resume file.

    Returns:
        The text content of the file.
    """
    if not os.path.exists(filepath):
        return f"Error: File not found at {filepath}"

    try:
        content = read_file_content(filepath)
        if not content:
            return "Error: Could not extract text from file or file is empty."
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"
