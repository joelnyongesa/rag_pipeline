"""
This module defines the response models for successful and error responses.
"""

from pydantic import BaseModel

class Success(BaseModel):
    """Model for successful responses."""
    success: bool = True
    message: str = "Data ingested successfully"
    data: dict = {}

class Error(BaseModel):
    """Model for error responses."""
    success: bool = False
    message: str = "Error ingesting data"
    error: str = ""
