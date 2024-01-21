import sys
sys.path
sys.path.append("/root/")  
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import ArxivLoader
from Docqa.text_spliter import ChineseTextSplitter
from bigdl.llm.langchain.embeddings import TransformersEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.chat_vector_db.prompts import QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from bigdl.llm.langchain.llms import TransformersLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

print("loading documents....")
loader = ArxivLoader(query="2304.04912")
pages = loader.load()
pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=165)
pdf_text = pdf_splitter.split_documents(pages)
print("documents loading successfully!")

embeddings = TransformersEmbeddings.from_model_id(model_id="/data/vicuna-7b-v1.5")
docsearch = FAISS.from_documents(pdf_text, embeddings).as_retriever()

llm = TransformersLLM.from_model_id(
        model_id="/data/vicuna-7b-v1.5",
        model_kwargs={"temperature": 0, "max_length": 1024, "trust_remote_code": True},
    )

doc_chain = load_qa_chain(
    llm, chain_type="stuff", prompt=QA_PROMPT
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=docsearch, memory=memory)

while True:
    query= input(">>>")
    result = qa.run(question=query)



