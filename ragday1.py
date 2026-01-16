
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_experimental.text_splitter import SemanticChunker

# 1. Read document
with open(r"D:\python\python\tutorial\cat_facts.txt", "r", encoding="utf-8") as file:
    document = file.read()

# 2. Initialize embeddings
embeddings = OllamaEmbeddings(model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest")

# 3. Chunk text with embeddings and size limits
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)
chunks = splitter.split_text(document)

# Filter and truncate chunks to safe size
MAX_CHUNK_LENGTH = 2000
chunks = [
    chunk[:MAX_CHUNK_LENGTH] if len(chunk) > MAX_CHUNK_LENGTH else chunk 
    for chunk in chunks 
    if len(chunk.strip()) > 50
]

print(f"Total chunks created: {len(chunks)}")
print(f"Max chunk length: {max(len(chunk) for chunk in chunks)}")

# 4. Create vectorstore and LLM
vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
llm = Ollama(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

# 5. Create retriever and memory
retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Fixed: lowercase
    return_messages=True,
    output_key="answer"
)

# 6. Create QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# 7. Query using the chain (this is the key fix)
question = input("Write a Question to LLM: ")
response = qa_chain({"question": question})
print(response["answer"])

