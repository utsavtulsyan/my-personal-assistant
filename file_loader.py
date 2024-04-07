from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

directory_loader = DirectoryLoader('mydocs')
raw_documents = []
print('Loading docs...')
raw_documents = directory_loader.load()
print('Loading completed!')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(raw_documents)

print('Splitting completed!')

db = FAISS.from_documents(split_documents, OllamaEmbeddings())

print('Indexing completed!')
db.save_local('./db')
print('Saved index to local db!')