import os
import json
import asyncio
from openai import OpenAI
from astra_assistants import patch
from astra_assistants.astra_assistants_manager import AssistantManager
from appdirs import user_config_dir
from astra_assistants.mcp_openai_adapter import MCPRepresentationStdio


APP_NAME = "agentd"
APP_AUTHOR = "phact"

# Determine the correct configuration directory and file path.
config_dir = user_config_dir(APP_NAME, APP_AUTHOR)
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, "config.json")

# Default configuration structure
default_config = {
    "assistant_id": None,
    "thread_id": None
}

config=None
assistant_manager = None
thread_id = None
assistant_id = None
continue_thread = False

def load_config():
    """Load configuration from the JSON file, or return defaults if missing."""
    global config
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            return
        except Exception as e:
            print(f"Error loading config: {e}")
    # If the file doesn’t exist or an error occurs, return a copy of the defaults.
    config = default_config.copy()

def save_config():
    """Save the configuration to the JSON file."""
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

async def run_thread(prompt, thread_id):
    """
    Run a thread (conversation) with the assistant.
    
    Replace or adjust this function depending on how your assistant_manager
    interacts with the API. This function is expected to return a dictionary
    that contains at least a 'thread_id' key when a new thread is created.
    """
    result = await assistant_manager.run_thread(content=prompt, thread_id=thread_id)
    return result

def init_manager(instructions: str):
    print("Initializing the application...")

    # Load existing configuration (if any)
    global config, assistant_manager, continue_thread

    mcps = [
        MCPRepresentationStdio(
            type="stdio",
            command="uv",
            arguments=[
                "run",
                "--with",
                "../mcp-google-docs",
                "server",
                "--creds-file-path",
                "../mcp-google-docs/.auth/client_secret_1049158366095-o78hnepquu77t0uf3gr7q5ik887a6tvv.apps.googleusercontent.com.json",
                "--token-path",
                "../mcp-google-docs/.auth/token"
            ]
        )
    ]

    load_config()
    if continue_thread:
        thread_id = config.get("thread_id", None)
    assistant_id = config.get("assistant_id", None)

    # Create or patch your OpenAI client
    client = patch(OpenAI())
    
    # Create the AssistantManager with the assistant's ID.
    print(assistant_id)
    assistant_manager = AssistantManager(
        client=client,
        instructions=instructions,
        assistant_id=assistant_id,
        mcp_represenations=mcps
    )
    if not assistant_id:
        config["assistant_id"] = assistant_manager.assistant.id
        save_config()
        print("New assitant created with assistant_id:", assistant_id)
    
def run(prompt: str):
    global thread_id
    new_thread = not thread_id
    if new_thread:
        thread_id = config.get("thread_id")
    result = asyncio.run(run_thread(prompt, thread_id))
    if new_thread:
        config["thread_id"] = assistant_manager.thread.id
        save_config()
        print("New thread started with thread_id:", thread_id)
    print(result)
    print(result['text'])
    for tool in assistant_manager.tools:
        if tool.name == "read-comments":
            model = tool.get_model()
            # model is a BaseModel so you could introspect...
            model.model_fields.values()
            # and then to call the tool pass it a dict
            result = tool.call(model(title="title").dump_model())

if __name__ == "__main__":
    instructions = "You are a long running agent that assists with google docs"
    prompt = "create a new doc"
    init_manager(instructions)
    run(prompt)
