from fastapi import FastAPI
from src.routes.ingest_data import ingest_router

app = FastAPI()

app.include_router(ingest_router, prefix="/api/v1")