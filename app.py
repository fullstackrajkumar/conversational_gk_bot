from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

file_path = "./data/paper1.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

# For making embeddings

# vector_db = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory="./db"
# )

# for getting the vector store

vector_db = Chroma(
    persist_directory="./db",
    embedding_function=embeddings
)

llm = Ollama(model="llama3")

retriever = vector_db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever,return_source_documents=True)

while True:
    question = input("\nPlease ask your question : ")
    print("Fetching your answer ...............................................................")
    answer = qa_chain.invoke(question)
    print(answer['result'])