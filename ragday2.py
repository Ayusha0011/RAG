from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# 1️⃣ Load PDF
loader = PyPDFLoader(r"D:\RAG\Read.pdf")
docs = loader.load()

# 2️⃣ Split into parent chunks (high-level context)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
parent_chunks = parent_splitter.split_documents(docs)

# 3️⃣ Split each parent into child chunks (fine-grained)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50
)
child_chunks = []
for doc in parent_chunks:
    child_chunks.extend(child_splitter.split_documents([doc]))

# 4️⃣ Create embeddings
embeddings = OllamaEmbeddings(
    model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"
)

# 5️⃣ Store child chunks in Chroma
vectorstore = Chroma(
    collection_name="hierarchical_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
vectorstore.add_documents(child_chunks)

# 6️⃣ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":5})  # fetch top 5 relevant chunks

# 7️⃣ LLM
llm = Ollama(
    model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
    temperature=0.1
)

# 8️⃣ QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# 9️⃣ Query
query = "Explain the main concept discussed in the document"
result = qa_chain(query)

# 10️⃣ Output
print("ANSWER:\n", result["result"])
print("\nSOURCES:")
for doc in result["source_documents"]:
    print(doc.metadata)
