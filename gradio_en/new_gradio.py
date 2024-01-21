import gradio as gr


# with gr.Blocks() as demo:
#     gr.HTML("""<h1 align="center">🤗Welcome Documents Assistant🤗</h1>""")
    

#     with gr.Row():
#         with gr.Column(scale=4):
#             chatbot = gr.Chatbot(label="ChatGLM-6B bot",value=[],elem_id='chatbot',height=550)
#             with gr.Column():
#                 user_input = gr.Text(show_label=False, placeholder="Input...", container=False)
#             with gr.Column(min_width=32, scale=1):
#                 with gr.Row():
#                     submit_btn = gr.Button("Submit")
#                     emptyBtn = gr.Button("Clear History")

#         with gr.Column(scale=1):
#             file_output = gr.File(label="Your PDFs")
#             btn = gr.UploadButton("📁 Upload a PDF(s)",
#                                       file_types=[".pdf"],
#                                       file_count="multiple")
#             analysis = gr.Button('构建向量数据库!')
            
#             max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
#             top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
#             temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


    # def user(query, history):
    #     return "", history + [[parse_text(query), ""]]


    # submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
    #     predict, [chatbot, max_length, top_p, temperature], chatbot
    # )
    # emptyBtn.click(lambda: None, None, chatbot, queue=False)

import gradio as gr
import time

def my_function(x, progress=gr.Progress()):
    progress(0, desc="开始...")
    time.sleep(1)
    for i in progress.tqdm(range(100)):
        time.sleep(0.1)
    return x

gr.Interface(my_function, gr.Textbox(), gr.Textbox()).queue().launch()
