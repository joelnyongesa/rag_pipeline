from pydantic import BaseModel

class Success(BaseModel):
    success: bool = True
    message: str = "Data ingested successfully"
    data: dict = {}

class Error(BaseModel):
    success: bool = False
    message: str = "Error ingesting data"
    error: str = ""