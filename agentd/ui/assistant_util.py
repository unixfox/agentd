from astra_assistants.astra_assistants_manager import AssistantManager

def create_manager(instructions: str, model: str = "gpt-4o") -> AssistantManager:
    """
    Create and return an AssistantManager.
    """
    return AssistantManager(instructions=instructions, model=model)

def list_threads(manager: AssistantManager):
    """
    Retrieve threads from the API via manager.client.beta.threads.list().
    If that call returns an empty list or fails, fallback to the manager's thread.
    """
    try:
        threads = manager.client.beta.threads.list()
        if not threads:
            raise Exception("No threads found.")
    except Exception as e:
        threads = [manager.thread]
    return threads.data

def list_messages(manager: AssistantManager, thread_id: str):
    """
    Retrieve the messages for a given thread by its ID.
    """
    return manager.client.beta.threads.messages.list(thread_id)
