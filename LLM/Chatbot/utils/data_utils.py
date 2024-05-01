from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings



# retrive in long text

################################

def file_to_corpus(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus = f.readlines()
    else:
        raise ValueError('File type not supported')
    return corpus

def corpus_to_documents(corpus):
    documents = [Document(page_content=cp) for cp in corpus]
    return documents

def retrieve_documents(documents, question, k=4, model='text-embedding-ada-002'):
    embedding = OpenAIEmbeddings(model=model) 
    documents_db = FAISS.from_documents(documents, embedding)
    retrieval_documents = documents_db.max_marginal_relevance_search(question, k=k)
    return retrieval_documents
            
#################################
