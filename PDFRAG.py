from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

print("Starting RAG pipeline...")

# Step 1: Load the PDF document
print("Loading PDF document...")
loader = PyPDFLoader(r"D:\RAG\Read.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages from PDF")

# Step 2: Hierarchical Chunking
print("Creating parent chunks...")
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)
parent_chunks = parent_splitter.split_documents(docs)
print(f"Created {len(parent_chunks)} parent chunks")

print("Creating child chunks...")
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)

# Initialize lists and mapping dictionary
child_chunks = []
parent_child_map = {}

# Split parent chunks into child chunks and create parent-child mapping
for i, parent in enumerate(parent_chunks):
    children = child_splitter.split_documents([parent])
    
    # Add parent index to each child's metadata for stable mapping
    for j, child in enumerate(children):
        child.metadata['parent_index'] = i
        child.metadata['child_index'] = len(child_chunks) + j
    
    child_chunks.extend(children)
    # Store parent by its index
    parent_child_map[i] = parent

print(f"Created {len(child_chunks)} child chunks")

# Step 3: Create embeddings for child chunks
print("Initializing embeddings model...")
embeddings = OllamaEmbeddings(
    model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"
)

# Step 4: Store child chunks in a vector store for similarity search
print("Creating vector store and adding documents...")
vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=embeddings,
    persist_directory=r"D:\RAG\chroma_db"
)

try:
    vectorstore.add_documents(child_chunks)
    print(f"Successfully added {len(child_chunks)} documents to vector store")
    print(f"Vector database stored at: D:\\RAG\\chroma_db")
except Exception as e:
    print(f"Error adding documents to vector store: {e}")
    exit(1)

# Step 5: Define a custom hierarchical retriever
def custom_hierarchical_retriever(query, k=3):
    # Search for the top-k most similar child chunks
    results = vectorstore.similarity_search(query, k=k)
    
    # Collect corresponding parent chunks without duplicates
    parents_to_return = []
    seen = set()
    
    for child in results:
        parent_index = child.metadata.get('parent_index')
        if parent_index is not None and parent_index not in seen:
            parents_to_return.append(parent_child_map[parent_index])
            seen.add(parent_index)
    
    return parents_to_return

# Step 6: Initialize LLM
print("Initializing LLM...")
llm = Ollama(
    model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
    temperature=0.1
)

# Step 7: Define RAG query function
def rag_query(question):
    # Retrieve relevant parent chunks
    relevant_parents = custom_hierarchical_retriever(question, k=3)
    
    # Combine parent chunks into context
    context = "\n\n".join([parent.page_content for parent in relevant_parents])
    
    prompt = f"""You are a well-trained document analyst who reads and describes document content accurately.

Your task:
- Provide unambiguous and concise answers to user queries
- Answer ONLY based on the information in the provided context
- Do not make assumptions or add information not present in the context
- If the answer is not in the context, clearly state that

Context:
{context}

Question: {question}

Instructions: Carefully analyze the context above, extract the relevant facts, and provide a clear, accurate answer in your own words.

Answer:"""
    
    answer = llm.invoke(prompt)
    return answer

print("\nRAG pipeline ready!")
print("=" * 50)

# Step 8: Interactive query loop
while True:
    ask = input("\nEnter your question (or type 'exit' to quit): ")
    if ask.lower() == "exit":
        print("Exiting RAG pipeline. Goodbye!")
        break
    else:
        print("\nProcessing your query...")
        try:
            answer = rag_query(ask)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"Error processing query: {e}")


