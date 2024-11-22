import os
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader

os.environ['GROQ_API_KEY'] = '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'

sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0]

    def __call__(self, texts):
        # Allow the object to be used as a callable for document embedding
        return self.embed_documents(texts)


class DocumentInput(BaseModel):
    question: str = Field()

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

def process_query_with_documents(doc1_path, doc2_path, user_query):
    print(f"Processing documents:\nDoc1: {doc1_path}\nDoc2: {doc2_path}")
    print(f"Query: {user_query}")

    doc1 = os.path.splitext(os.path.basename(doc1_path))[0]
    doc2 = os.path.splitext(os.path.basename(doc2_path))[0]

    tools = []
    files = [
        {"name": doc1, "path": doc1_path},
        {"name": doc2, "path": doc2_path},
    ]

    embeddings = SentenceTransformerEmbeddings(model=sentence_transformer_model)
    print("Embeddings instance created successfully.")

    for file in files:
        print(f"Processing file: {file['path']}")
        loader = PyPDFLoader(file["path"])
        pages = loader.load_and_split()
        print(f"Number of pages loaded: {len(pages)}")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        print(f"Number of chunks: {len(docs)}")

        retriever = FAISS.from_documents(docs, embeddings).as_retriever(search_type="similarity", search_kwargs={"k": 5})
        print("Retriever Type:", type(retriever))

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,  # Include source document details
        )

        tools.append(
            Tool(
                args_schema=DocumentInput,
                name=file["name"],
                description=f"Useful for answering questions about {file['name']}",
                func=qa_chain,
            )
        )

    print("Initializing agent...")
    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
    )

    print("Sending query to the agent...")
    response_dict = agent({"input": user_query, "return_source_documents": True})
    print("Agent Response:", response_dict)

    result = response_dict["output"]

    if "source_documents" in response_dict:
        source_details = "\n".join([doc.page_content for doc in response_dict["source_documents"]])
        result += f"\n\nSource Details:\n{source_details}"

    return result
