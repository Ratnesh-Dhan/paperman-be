import fitz # PyMuPDF
import re

class Extractor:
    def extract_pdf_text(self, file_path):
        doc = fitz.open(file_path)
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "page": i+1,
                "text": text
            })
        return pages

    def split_into_sections(self, pages):
        sections = []
        current_section = {"title": "Unknown", "content": "", "page": ""}

        for page in pages:
            lines = page["text"].split("\n")

            for line in lines:
                line = line.strip()

                # Detect headings (ALL CAPS or numbered)
                if re.match(r"^([A-Z][A-Z\s]{3,}|(\d+\.\s.*))$", line):
                    if current_section["content"]:
                        sections.append(current_section)

                    current_section = {
                        "title": line,
                        "content": ""
                    }
                else: 
                    current_section["content"] += line + " "
            
        if current_section["content"]:
            sections.append(current_section)
        return sections

    def chunk_section(self, section, chunk_size=500, overlap=100):
        words = section["content"].split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "section": section["title"]
                }
            })
        return chunks