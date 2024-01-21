import gradio as gr
import PyPDF2 # pdf reader
import time
from pypdf import PdfReader
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import pipeline, AutoTokenizer, TextIteratorStreamer, LlamaTokenizer
# from langchain.llms.huggingface_pipeline import HuggingFacePipeline 
from bigdl.llm.langchain.embeddings import TransformersEmbeddings
from bigdl.llm.langchain.llms import TransformersLLM
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import threading

print("-"*20+"loading llm and embeddings......"+"-"*20)
embeddings = TransformersEmbeddings.from_model_id(model_id="/data/chatglm3-6B")
llm = TransformersLLM.from_model_id(
        model_id="/data/chatglm3-6B",
        model_kwargs={"temperature": 0, "max_length": 8192, "trust_remote_code": True})
print("-"*20+"loading llm and embeddings successfully!"+"-"*20)

tokenizer = LlamaTokenizer.from_pretrained('/data/chatglm3-6B')

# def load_llm():
#     # Loads the  DeciLM-7B-instruct llm when called
#     llm = TransformersLLM.from_model_id(
#         model_id="/data/vicuna-7b-v1.5",
#         model_kwargs={"temperature": 0, "max_length": 1024, "trust_remote_code": True},
#     )
    
#     return llm
qa_chain = None


def add_text(history, text):
      # Adding user query to the chatbot and chain
      # use history with curent user question
      if not text:
          raise gr.Error('Enter text')
      history = history + [(text, '')]
      return history


def upload_file(files):
      # Loads files when the file upload button is clicked
      # Displays them on the File window
      # print(type(file))
      file_path = [file.name for file in files]
      return file_path

def split_and_embedding_file(files):
      print("-"*20+"loading documents...."+"-"*20)
      pdf_text = ""
      for file in files:
        pdf = PyPDF2.PdfReader(file.name)
        for page in pdf.pages:
            pdf_text += page.extract_text()
      print("-"*20+"loading successfully!"+"-"*20)


      # split into smaller chunks
      print("-"*20+"spliting texts....."+"-"*20)
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
      splits = text_splitter.create_documents([pdf_text])
      print("-"*20+"spliting successfully!"+"-"*20)

      # create a FAISS vector store db
      # embedd the chunks and store in the db
      print("-"*20+"embedding...."+"-"*20)
      vectorstore_db = FAISS.from_documents(splits, embeddings)
      retriever = vectorstore_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
      print("-"*20+"embedding successfully!"+"-"*20)
      

      custom_prompt_template = """
      你是一个优秀的AI助手，现在给你提供了一些材料，请你根据这些材料回答用户提出的问题。如果你无法根据提供的信息回答问题，请直接回答不知道。\n
      不要尝试编造答案。
      材料: {context}
      历史记录: {history}
      问题: {question}

      有用的回答:
      """
      prompt = PromptTemplate(template=custom_prompt_template, input_variables=["question", "context", "history"])
      print("++++++++++++++++++++++++++++++")
      # set QA chain with memory
      qa_chain_with_memory = RetrievalQA.from_chain_type(
          llm=llm, chain_type='stuff', return_source_documents=True,
          retriever=retriever, 
          chain_type_kwargs={
              "verbose": False,
              "prompt": prompt,
              "memory": ConversationBufferMemory(
                input_key="question",
                memory_key="history",
                return_messages=True)})
      # get qa_chain
      global qa_chain
      qa_chain = qa_chain_with_memory
      return qa_chain_with_memory
   
   

def generate_bot_response(history,query, btn):
    """Function takes the query, history and inputs from the qa chain when the submit button is clicked
    to generate a response to the query"""

    if not btn:
        raise gr.Error(message='Upload a PDF')

    if qa_chain is not None:
        
        qa_chain_with_memory = qa_chain # run the qa chain with files from upload
        print("-"*20+"got qa_chain"+"-"*20)

    # streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    # # def run_qa_chain(message):
    # #     qa_chain_with_memory.run(message)
        
    # chain_streaming_thread = threading.Thread(target=qa_chain_with_memory.run, kwargs={"inputs":{"query":query}})
    # chain_streaming_thread.start()
    # for new_token in 

    bot_response = qa_chain_with_memory({"query": query})
    # simulate streaming
    for char in bot_response['result']:
          history[-1][-1] += char
          time.sleep(0.05)
          yield history,''




# The GRADIO Interface
with gr.Blocks() as demo:
    with gr.Row():
            with gr.Row():
              # Chatbot interface
              chatbot = gr.Chatbot(label="Vicuna-7B-v1.5 bot",
                                   value=[],
                                   elem_id='chatbot')
            with gr.Row():
              # Uploaded PDFs window
              file_output = gr.File(label="Your PDFs")
              

              with gr.Column():
                # PDF upload button
                btn = gr.UploadButton("📁 Upload a PDF(s)",
                                      file_types=[".pdf"],
                                      file_count="multiple")
              
              with gr.Column():
                analysis = gr.Button('Construct Vectors DB!')

    with gr.Column():
        with gr.Column():
          # Ask question input field
          txt = gr.Text(show_label=False, placeholder="Enter question")

        with gr.Column():
          # button to submit question to the bot
          submit_btn = gr.Button('Ask')

    # Event handler for uploading a PDF
    # print("*******************")
    btn.upload(fn=upload_file, inputs=[btn], outputs=[file_output])
    # print("===========================")

    analysis.click(fn = split_and_embedding_file,  inputs=[btn])

    # Event handler for submitting text question and generating response
    submit_btn.click(
        fn= add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
        ).success(
          fn=generate_bot_response,
          inputs=[chatbot, txt, btn],
          outputs=[chatbot, txt]
        ).success(
          fn=upload_file,
          inputs=[btn],
          outputs=[file_output]
        )
    

demo.queue()
demo.launch()