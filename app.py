import os
import json
import asyncio
from openai import OpenAI
from astra_assistants import patch
from astra_assistants.astra_assistants_manager import AssistantManager
from appdirs import user_config_dir

APP_NAME = "mcp-maker"
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

def load_config():
    """Load configuration from the JSON file, or return defaults if missing."""
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
    # If the file doesn’t exist or an error occurs, return a copy of the defaults.
    return default_config.copy()

def save_config(config):
    """Save the configuration to the JSON file."""
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

async def run_thread(assistant_manager, prompt):
    """
    Run a thread (conversation) with the assistant.
    
    Replace or adjust this function depending on how your assistant_manager
    interacts with the API. This function is expected to return a dictionary
    that contains at least a 'thread_id' key when a new thread is created.
    """
    result = await assistant_manager.run_thread(content=prompt)
    return result

def init():
    print("Initializing the application...")

    # Load existing configuration (if any)
    config = load_config()
    
    # Create or patch your OpenAI client
    client = patch(OpenAI())
    
    # -------------------------------
    # Handle the assistant_id
    # -------------------------------
    if not config.get("assistant_id"):
        print("No assistant_id found, creating a new assistant.")
        # Create a new assistant.
        # Adjust the parameters/method as required by your API.
        assistant = client.beta.assistants.create(model="gpt-4o-mini")  
        config["assistant_id"] = assistant.id
        save_config(config)
    else:
        print("Found existing assistant_id, retrieving assistant.")
        assistant = client.beta.assistants.retrieve(config["assistant_id"])
    
    print("Assistant:", assistant)
    
    # Create the AssistantManager with the assistant's ID.
    assistant_manager = AssistantManager(
        client=client,
        assistant_id=assistant.id,
        instructions=None
    )
    
def run():
    prompt = "draw a cat"
    if not config.get("thread_id"):
        result = asyncio.run(run_thread(assistant_manager, prompt))
        
        thread_id = result.get("thread_id")
        if thread_id:
            config["thread_id"] = thread_id
            save_config(config)
            print("New thread started with thread_id:", thread_id)
        else:
            print("Warning: No thread_id returned from run_thread.")
    else:
        # If a thread_id exists, you might choose to continue that conversation.
        print("Using existing thread_id:", config["thread_id"])
        result = asyncio.run(run_thread(assistant_manager, prompt))
        print("Thread response:", result)
    
    print("Final Assistant ID:", assistant.id)
    print("Final Thread ID:", config.get("thread_id"))

if __name__ == "__main__":
    init()
    run()
