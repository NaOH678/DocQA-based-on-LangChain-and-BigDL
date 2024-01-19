from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from Docqa.text_spliter import ChineseTextSplitter


def load_file(filepath:str):

    
    if filepath.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path=filepath)
        text_splitter = ChineseTextSplitter(chunk_size=650, chunk_overlap=0)
        input_text = loader.load_and_split(text_splitter=text_splitter)
    elif filepath.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path=filepath)
        text_splitter = ChineseTextSplitter(chunk_size=650, chunk_overlap=0, pdf=True)
        input_text = loader.load_and_split(text_splitter=text_splitter)
    elif filepath.lower().endswith('.md'):
        print('敬请期待')

    else:
        loader = TextLoader(filepath, encoding="utf8")
        textsplitter = RecursiveCharacterTextSplitter(chunk_size=650,
                                         chunk_overlap=0,
                                         separators=["\n\n", "\n", " ", ""],
                                         length_function=len)
        input_text = loader.load_and_split(text_splitter=textsplitter)

    return input_text






