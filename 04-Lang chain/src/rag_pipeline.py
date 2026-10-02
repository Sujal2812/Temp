from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("../data/documents/company_policy.pdf")

documents = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# Embeddings
embedding = HuggingFaceEmbeddings()

# Store in ChromaDB
vectordb = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="../data/chroma_db"
)

vectordb.persist()

# Search query
query = "What is company leave policy?"

results = vectordb.similarity_search(query)

print("\n===== SEARCH RESULTS =====\n")

for result in results:
    print(result.page_content)
    print("-" * 50)