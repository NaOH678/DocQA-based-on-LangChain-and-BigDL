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

print("-"*20+"loading embeddings......"+"-"*20)

def get_model_kwargs():
     return  dict({"temperature":0.5, "max_length":8192, 'top_p':0.6, "trust_remote_code": True})
     
embeddings = TransformersEmbeddings.from_model_id(model_id="/data/chatglm3-6B")
# model_kwargs = {"temperature":temperature, "max_length":max_length, 'top_p':top_p}
llm = TransformersLLM.from_model_id(
        model_id="/data/chatglm3-6B",
        model_kwargs=get_model_kwargs())

print("-"*20+"loading embeddings successfully!"+"-"*20)

tokenizer = LlamaTokenizer.from_pretrained('/data/chatglm3-6B')

# def load_llm():
#     # Loads the  DeciLM-7B-instruct llm when called
#     llm = TransformersLLM.from_model_id(
#         model_id="/data/vicuna-7b-v1.5",
#         model_kwargs={"temperature": 0, "max_length": 1024, "trust_remote_code": True},
#     )
    
#     return llm
retriever = None


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

def split_and_embedding_file(files, progress=gr.Progress()):
    #   progress(0, desc="开始构建向量数据库...")
    #   for i in progress.tqdm(range(100)):
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
        retriever_ = vectorstore_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
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
        # print("++++++++++++++++++++++++++++++")
        # set QA chain with memory
        # qa_chain_with_memory = RetrievalQA.from_chain_type(
        #     llm=llm, chain_type='stuff', return_source_documents=True,
        #     retriever=retriever, 
        #     chain_type_kwargs={
        #         "verbose": False,
        #         "prompt": prompt,
        #         "memory": ConversationBufferMemory(
        #             input_key="question",
        #             memory_key="history",
        #             return_messages=True)})
        # get qa_chain
        global retriever
        retriever = retriever_
        return "构建完成✅"
   
   

def generate_bot_response(history,query, btn):
    """Function takes the query, history and inputs from the qa chain when the submit button is clicked
    to generate a response to the query"""

    if not btn:
        raise gr.Error(message='Upload a PDF')
    
    custom_prompt_template = """
        你是一个优秀的AI助手，现在给你提供了一些材料，请你根据这些材料回答用户提出的问题。如果你无法根据提供的信息回答问题，请直接回答不知道。\n
        不要尝试编造答案。
        材料: {context}
        历史记录: {history}
        问题: {question}

        有用的回答:
        """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["question", "context", "history"])
    

    if retriever is not None:
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
        
         # run the qa chain with files from upload
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


          
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">🤗Welcome Documents Assistant🤗</h1>""")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="ChatGLM-6B bot",value=[],elem_id='chatbot',height=550)
            with gr.Column():
                txt = gr.Text(show_label=False, placeholder="Input...", container=False)
            with gr.Column(min_width=32, scale=1):
                with gr.Row():
                    submit_btn = gr.Button("提交")
                    emptyBtn = gr.Button("清除记录")

        with gr.Column(scale=1):
            file_output = gr.File(label="你的PDF文件")
            btn = gr.UploadButton("📁 上传PDF文件", file_types=[".pdf"], file_count="multiple")
            analysis = gr.Button('构建向量数据库!')
            xx = gr.Text(label= '向量数据库状态')
            
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)    
            # cons_model = gr.Button("构建模型")


    # Event handler for uploading a PDF
    # print("*******************")
    btn.upload(fn=upload_file, inputs=[btn], outputs=[file_output])
    # print("===========================")

    analysis.click(fn = split_and_embedding_file,  inputs=[btn], outputs=[xx])
    

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
    emptyBtn.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()