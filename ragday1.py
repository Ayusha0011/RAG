from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms setup.exeimport Ollama
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# 1️⃣ Load PDF
loader = PyPDFLoader(r"D:\RAG\Read.pdf")
docs = loader.load()

# 2️⃣ Parent chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
parent_chunks = parent_splitter.split_documents(docs)

# 3️⃣ Child chunks
child_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
child_chunks = []
for doc in parent_chunks:
    child_chunks.extend(child_splitter.split_documents([doc]))

# 4️⃣ Embeddings
embeddings = OllamaEmbeddings(
    model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"
)

# 5️⃣ Chroma vector store
vectorstore = Chroma(
    collection_name="hierarchical_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
vectorstore.add_documents(child_chunks)

# 6️⃣ Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 7️⃣ LLM
llm = Ollama(
    model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
    temperature=0.1
)

# 8️⃣ QA Chain - FIXED VERSION
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 9️⃣ Query
query = "Explain the main concept discussed in the document"
result = qa_chain.invoke({"query": query})

print("ANSWER:\n", result["result"])
print("\nSOURCE DOCUMENTS:")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:200])