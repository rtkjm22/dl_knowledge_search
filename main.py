from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr

# テキストファイル読み込み
loader = DirectoryLoader("./documents", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# テキスト分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Embedding作成
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(docs, embedding)
vectordb.save_local("./embeddings")

# プロンプトテンプレート定義
custom_prompt = PromptTemplate.from_template("""
あなたは社内規約の専門アシスタントです。
以下のドキュメントを参考に、質問に対して正確かつ丁寧に日本語で答えてください。

# ドキュメント:
{context}

# 質問:
{question}

# 回答（敬語で、わかりやすく）:
""")

# LLMモデル読み込み
llm = GPT4All(
    model="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    backend="llama",
    max_tokens=256,
    verbose=True
)

# ベクトル検索のRetriever（文書数制限）
retriever = vectordb.as_retriever(search_kwargs={"k": 1})

# QAチェーン
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

# チャットUI
def chat_fn(message):
    return qa.run(message)

gr.Interface(fn=chat_fn, inputs="text", outputs="text", title="社内ナレッジチャットボット💼").launch()
