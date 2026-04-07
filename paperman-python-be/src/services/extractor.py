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
        current_section = {"title": "Unknown", "content": "", "page": None}

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
                        "content": "",
                        "page": page["page"]
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
                    "section": section["title"],
                    "page": section["page"]
                }
            })
        return chunks

    def extract_blocks(self, file_path):
        doc = fitz.open(file_path)
        blocks = []

        for page_num, page in enumerate(doc, 1):
            page_blocks = page.get_text("blocks")

            for block in page_blocks:
                text = block[4].strip()

                # skip tiny junk blocks
                if len(text) < 30:
                    continue

                # normalize whitespace
                text = " ".join(text.split())

                blocks.append({
                    "page": page_num,
                    "text": text
                })

        return blocks

    def chunk_blocks(self, blocks, chunk_size=400, overlap=80):
        chunks = []
        current_words = []
        current_pages = set()

        for block in blocks:
            words = block["text"].split()

            # if adding this block exceeds chunk size
            if len(current_words) + len(words) > chunk_size:
                if current_words:
                    chunks.append({
                        "text": " ".join(current_words),
                        "metadata": {
                            "pages": sorted(current_pages)
                        }
                    })

                # keep overlap words
                current_words = current_words[-overlap:] if overlap < len(current_words) else current_words
                current_pages = {block["page"]}

            current_words.extend(words)
            current_pages.add(block["page"])

        # final leftover chunk
        if current_words:
            chunks.append({
                "text": " ".join(current_words),
                "metadata": {
                    "pages": sorted(current_pages)
                }
            })

        return chunks