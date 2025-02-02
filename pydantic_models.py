from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    llama3_2_1b = "llama3.2:1b"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.llama3_2_1b)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName


