from llama_index.core.schema import TextNode
from src.services.extractor import Extractor

class Nodes:
    def create_nodes_from_chunks(self, chunks):
        nodes = []

        for chunk in chunks:
            node = TextNode(
                text=chunk["text"],
                metadata=chunk["metadata"]
            )
            nodes.append(node)

        return nodes
    def create_nodes(self, sections):
        extractor = Extractor()
        nodes = []
        for section in sections:
            chunks = extractor.chunk_section(section)

            for chunk in chunks:
                node = TextNode(
                    text=chunk["text"],
                    metadata=chunk["metadata"]
                )
                nodes.append(node)
        return nodes