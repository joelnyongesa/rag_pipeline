"""
This module serves as the entry point for the FastAPI application.

It initializes the FastAPI instance and includes routers for different API endpoints (eg. query).
"""

from fastapi import FastAPI
from src.routes.ingest_data import ingest_router

app = FastAPI()

app.include_router(ingest_router, prefix="/api/v1")
