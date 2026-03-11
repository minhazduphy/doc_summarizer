import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
load_dotenv()

# Load PDF
loader = PyPDFLoader("/workspaces/coursera_projects/doc_summarizer/Handbook Personnel Policy final 06.01.25.pdf")
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
docsearch = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)

# LLM
llm = ChatOpenAI(model="gpt-4.1-mini")

prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definately do not try to make up an answer.

{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)

# Retrieval QA
qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                           chain_type="stuff", 
                                           retriever=docsearch.as_retriever(), 
                                           memory = memory, 
                                           get_chat_history=lambda h : h, 
                                           return_source_documents=False)
history = []

query = "What is mobile policy?"
result = qa.invoke({"question":query}, {"chat_history": history})

history.append((query, result["answer"]))

query = "List points in it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])

history.append((query, result["answer"]))

query = "What is the aim of it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])

def qa():
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                               chain_type="stuff", 
                                               retriever=docsearch.as_retriever(), 
                                               memory = memory, 
                                               get_chat_history=lambda h : h, 
                                               return_source_documents=False)
    history = []
    while True:
        query = input("Question: ")
        
        if query.lower() in ["quit","exit","bye"]:
            print("Answer: Goodbye!")
            break
            
        result = qa({"question": query}, {"chat_history": history})
        
        history.append((query, result["answer"]))
        
        print("Answer: ", result["answer"])
        
if __name__ == "__main__":
    qa()