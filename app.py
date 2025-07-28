from flask import Flask, render_template, request, jsonify
from src.helper import download_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embedding = download_embedding_model()
index_name = "medical-rag"

doc_search = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)


retriever = doc_search.as_retriever(
    search_type="similarity",
    search_kwargs={ "k": 3}
)

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

range_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    input = msg.strip()
    print(f"User input: {input}")
    response = range_chain.invoke({"input": input})
    print(f"Response: {response['answer']}")
    return str(response['answer'])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)