from src.indexing.loader import DocumentLoader
from src.indexing.cleaner import DocumentCleaner
from src.indexing.loaders.pdf_loader import PDFLoader

def indexing_pipeline(file_path : str) :
    documents = DocumentLoader().load_data(PDFLoader(path=file_path))
    clean_documents = DocumentCleaner().clean_documents(documents=documents)

    return clean_documents


if __name__ == '__main__' :
    docs = indexing_pipeline('docs/giddiness.pdf')
    print(len(docs)) 
    for doc in docs :
        print('++++++file content++++++++++')
        print(doc.page_content)
        print('++++++file metadata++++++++++')
        print(doc.metadata)