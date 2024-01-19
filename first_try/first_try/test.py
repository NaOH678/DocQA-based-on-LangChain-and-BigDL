from bigdl.llm.langchain.llms import TransformersLLM
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

llm = TransformersLLM.from_model_id(
        model_id="/data/vicuna-7b-v1.5/",
        model_kwargs={"temperature": 70, "max_length": 2048, "trust_remote_code": True},
    )

template ="USER: {question}\nASSISTANT:"
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

conversation_chain = ConversationChain(
    # verbose=True,
    # prompt=prompt,
    llm=llm,
    memory=ConversationBufferMemory(),
    # llm_kwargs={"max_new_tokens": 256},
)
while True:
    query = input(">>>")
    if query == 'exit':
        break

    # result = llm_chain.run(query)
    result = conversation_chain.run(query)
    