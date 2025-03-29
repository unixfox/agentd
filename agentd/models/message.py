from pydantic import BaseModel
from typing import List, Literal, Optional


# Define a Pydantic model matching your Message schema.
class ContentItem(BaseModel):
    text: str
    type: Literal["input_text", "output_text", "summary_text"]


class Message(BaseModel):
    content: List[ContentItem]
    role: Literal["user", "system", "developer", "assistant"]
    status: Optional[Literal["in_progress", "completed", "incomplete"]]

def create_message(raw_item) -> Message:
    """
    Convert a raw message item into a Pydantic Message model.
    """
    content_texts = [ContentItem(text=item.text, type=item.type) for item in raw_item.content]
    return Message(content=content_texts, role=raw_item.role, status=raw_item.status)
