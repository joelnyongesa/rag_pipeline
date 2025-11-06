from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from src.models.response import Success, Error

from uuid import uuid4
from typing import List, Union
import os
from dotenv import load_dotenv

load_dotenv()

ingest_router = APIRouter()

ALLOWED_EXTENSIONS = [".pdf"]

COLLECTION_NAME = "documents"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

os.makedirs(DATA_DIR, exist_ok=True)

# Normalizing .env values to avoid issues like trailing quotes or slashes
_qdrant_url_raw = os.getenv("QDRANT_URL")
_qdrant_url = (
    _qdrant_url_raw.strip().strip("'").strip('"').rstrip("/") if _qdrant_url_raw else None
)

_api_key_raw = os.getenv("QDRANT_API_KEY")
_api_key = _api_key_raw.strip().strip('"').strip("'") if _api_key_raw else None

client = QdrantClient(url=_qdrant_url, api_key=_api_key)

# Create a collection, only if it does not already exist
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vector_config=VectorParams(size=3071, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large")
)

# Query model
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query to search the vector store")
    k:int = Field(5, ge=1, le=50, description="Number of top results to return")


# File ingestion endpoint
@ingest_router.post("/ingest", tags=["ingest"], response_model=Union[Success, Error], status_code=200)
async def ingest(files: List[UploadFile] = File(...)):
    try:
        results = []

        for uploaded_file in files:
            filename = uploaded_file.filename
            extension = os.path.splitext(filename)[1].lower()
            
            if extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"File type {extension} not allowed.")

            # Create a unique file path in src/data
            unique_filename = f"{uuid4()}_{filename}"
            file_path = os.path.join(DATA_DIR, unique_filename)

            # Save file
            with open(file_path, "wb") as f:
                content = await uploaded_file.read()
                f.write(content)

            # Load using the appropriate loader
            if extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif extension == ".txt":
                loader = TextLoader(file_path)
            else:
                raise HTTPException(status_code=400, detail=f"No loader available for {extension} files.")
            
            documents = loader.load()

            # Split and embed
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            uuids = [str(uuid4()) for _ in range(len(chunks))]
            response_ids = vector_store.add_documents(chunks, ids=uuids)

            results.append({"filename": filename, "ids": response_ids})

        return Success(message="Files ingested successfully", data={"files": results})
    
    except Exception as e:
        return Error(message="Error ingesting files", error=str(e))
    

# Query endpoint
@ingest_router.post("/query", tags=["query"], response_model=Union[Success, Error], status_code=200)
async def query_vector_store(request: QueryRequest):
    try:
        docs_and_scores = vector_store.similarity_search_with_score(request.query, k=request.k)
        results = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in docs_and_scores
        ]

        return Success(
            success=True,
            message="Query executed successfully",
            data={"results": results}
        )
    
    except Exception as e:
        return Error(message="Error executing query", error=str(e))