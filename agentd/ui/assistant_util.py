from typing import Optional, Literal, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from astra_assistants.astra_assistants_manager import AssistantManager
from openai._models import BaseModel
from openai.types.shared.metadata import Metadata
from openai.types.beta.thread import ToolResources
from openai.types.beta.threads.message import Message

def create_manager(instructions: str, model: str = "gpt-4o") -> AssistantManager:
    """
    Create and return an AssistantManager.
    """
    return AssistantManager(instructions=instructions, model=model)

class AugmentedThread(BaseModel):
    id: str
    """The identifier, which can be referenced in API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the thread was created."""

    metadata: Optional[Metadata] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    object: Literal["thread"]
    """The object type, which is always `thread`."""

    tool_resources: Optional[ToolResources] = None
    """
    A set of resources that are made available to the assistant's tools in this
    thread. The resources are specific to the type of tool. For example, the
    `code_interpreter` tool requires a list of file IDs, while the `file_search`
    tool requires a list of vector store IDs.
    """
    messages: list[str]
    """
    A list of [messages](https://platform.openai.com/docs/api-reference/messages).
    """

def list_messages(manager: AssistantManager, thread_id: str):
    """
    Retrieve the messages for a given thread by its ID.
    """
    return manager.client.beta.threads.messages.list(thread_id)

def get_thread_with_messages(manager: AssistantManager, raw_thread) -> AugmentedThread:
    """
    Retrieve messages for a given raw_thread and return an AugmentedThread instance.
    """
    # Assumes raw_thread has an 'id' attribute and a to_dict() method.
    messages = list_messages(manager, raw_thread.id).data
    if messages:
        print(messages)
    return AugmentedThread(
        **raw_thread.to_dict(),
        messages=messages
    )

def list_threads(manager: AssistantManager):
    """
    Retrieve threads from the API via manager.client.beta.threads.list().
    If that call returns an empty list or fails, fallback to the manager's thread.
    This version parallelizes the retrieval of messages for each thread.
    """
    try:
        raw_threads = manager.client.beta.threads.list().data
        threads = []
        # Adjust max_workers as needed based on your workload and system.
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks for each raw thread.
            future_to_raw = {
                executor.submit(get_thread_with_messages, manager, raw_thread): raw_thread
                for raw_thread in raw_threads
            }
            for future in as_completed(future_to_raw):
                try:
                    thread = future.result()
                    threads.append(thread)
                except Exception as exc:
                    print(f"Error processing thread {future_to_raw[future].id}: {exc}")
        if not threads:
            raise Exception("No threads found.")
    except Exception as e:
        # Fallback if threads retrieval fails.
        threads = [manager.thread]
    return threads
