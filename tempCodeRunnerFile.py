from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings  import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.llms import ollama
from langchain.chains import RetrievalQA

loader = PyPDFLoader(r"D:\RAG\Read.pdf")
docs = loader.load()

#For Parents
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200,
)
#Child 
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 350,
    chunk_overlap = 50,
)
embeddings = OllamaEmbeddings(
    model = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"
)
vectorstore  = Chroma(
    collection_name= "hierarchical_rag",
    embedding_function= embeddings,
    persist_directory="./chroma_db"
)
#Hierarchical Chunking starts
docstore = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore = vectorstore,
    docstore = docstore,
    child_splitter = child_splitter,
    parent_splitter = parent_splitter,
)

retriever.add_documents(docs)

llm = ollama(
    model ="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",\
    temperature = 0.1
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",# A method in LangChain for handling documents by inserting all the text from the documents into a single prompt ,The stuff method combines a list of documents into a single string and passes it, along with the user's query or instruction (e.g., "summarize the following text"), to the language model in a single API call. 
    return_source_documents=True,
)
query = "Explain the main concept discussed in the document"
result = qa_chain(query)
print("ANSWER:\n", result["result"])
print("\nSOURCES:")
for doc in result["source_documents"]:
    print(doc.metadata)