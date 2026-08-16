import re
from langchain_core.documents import Document

class DocumentCleaner :

    def clean_documents(self, documents: list[Document]) -> list[Document]:
        """Make Simple Clean For Documents"""
        for doc in documents:
            text = doc.page_content

            text = text.replace("\u200b", "")
            text = text.replace("\xa0", " ")

            # Remove extra spaces
            text = re.sub(r"[ \t]+", " ", text)

            # Remove excessive new lines
            text = re.sub(r"\n{3,}", "\n\n", text)

            # # Remove spaces around new lines
            # text = re.sub(r" *\n *", "\n", text)

            doc.page_content = text.strip()

        return documents