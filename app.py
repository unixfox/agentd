import ast
import os
import json
import asyncio
import time
from datetime import datetime, timezone

from tools.done_tool import DoneTool
from openai import OpenAI
from astra_assistants import patch
from astra_assistants.astra_assistants_manager import AssistantManager
from appdirs import user_config_dir
from astra_assistants.mcp_openai_adapter import MCPRepresentationStdio

APP_NAME = "agentd"
APP_AUTHOR = "phact"
MAX_ITERATIONS = 5

# Configuration file paths
config_dir = user_config_dir(APP_NAME, APP_AUTHOR)
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, "config.json")

default_config = {
    "assistant_id": None,
    "thread_id": None
}

config = None
assistant_manager : AssistantManager = None
done_manager : AssistantManager = None
thread_id = None
assistant_id = None

# Global caches to detect changes
last_modified_time = None

# Global tool cache
tool_cache = {}

new_comment_event = asyncio.Event()

async def async_input(prompt: str) -> str:
    # Wrap the blocking input in a thread
    return await asyncio.to_thread(input, prompt)

def load_config():
    global config
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            return
        except Exception as e:
            print(f"Error loading config: {e}")
    config = default_config.copy()

def save_config():
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

async def check_done():
    global thread_id, done_manager
    tool = DoneTool()
    result = await done_manager.run_thread(content="Has the last user request been completed?", thread_id=thread_id, tool=tool)
    arguments = result.get("arguments")
    #print("COT thoughts:")
    #print(result.get('text', 'No text response found.'))
    return result

async def chat(prompt):
    global thread_id
    result = await assistant_manager.run_thread(content=prompt, thread_id=thread_id)
    arguments = result.get("arguments")
    if arguments is not None:
        print(f"Tool Call: {arguments.__class__.__name__}({arguments}) \n->{result['output']}")

    print("Assistant response:")
    print(result.get('text', 'No text response found.'))
    return result

def cache_tools():
    """Caches tools by their names for quick lookup."""
    global tool_cache
    tool_cache = {}
    for tool in assistant_manager.tools:
        tool_cache[tool.mcp_tool.name] = tool

def init_done_manager():
    print("Initializing done manager...")
    global config, done_manager, done_assistant_id

    load_config()
    done_assistant_id = config.get("done_assistant_id", None)

    instructions = ("You are a supervisor agent, your job is to use the tool to determine if a task has"
                    "been completed."
                    "For a task to be completed it has reflect in the document itself not just the conversation.")

    client = patch(OpenAI())
    done_manager = AssistantManager(
        client=client,
        instructions=instructions,
        assistant_id=done_assistant_id,
        tools=[DoneTool()],
    )

    if not done_assistant_id:
        config["done_assistant_id"] = done_manager.assistant.id
        save_config()
        print("New done assistant created with assistant_id:", done_manager.assistant.id)

def init_manager(instructions: str):
    print("Initializing the application...")
    global config, assistant_manager, thread_id, assistant_id
    print(f"init manager {thread_id}")

    load_config()
    thread_id = config.get("thread_id", None)
    assistant_id = config.get("assistant_id", None)

    mcps = [
        MCPRepresentationStdio(
            type="stdio",
            command="uv",
            arguments=[
                "run",
                "--refresh",
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

    client = patch(OpenAI())
    assistant_manager = AssistantManager(
        client=client,
        model="gpt-4o",
        instructions=instructions,
        assistant_id=assistant_id,
        mcp_represenations=mcps
    )

    if not assistant_id:
        config["assistant_id"] = assistant_manager.assistant.id
        save_config()
        print("New assistant created with assistant_id:", assistant_manager.assistant.id)

    # Cache tools for faster lookup
    cache_tools()

def call_tool_with_introspection(tool, provided_params: dict):
    """
    Utility function that introspects a tool's model to build a valid payload.

    It inspects the tool's model fields (assumed to be a Pydantic model) to:
    - Filter the provided_params to only those keys that exist in the model.
    - Check that all required fields are present.
    - Instantiate the model and call the tool with the dumped payload.
    """
    model_cls = tool.get_model()  # Get the model class
    # Collect valid parameters by filtering based on the model's fields
    valid_fields = set(model_cls.model_fields.keys())
    valid_params = {k: v for k, v in provided_params.items() if k in valid_fields}

    # Check for missing required fields
    for field_name, field_info in model_cls.model_fields.items():
        if field_info.is_required() and field_name not in valid_params:
            raise ValueError(f"Missing required parameter: {field_name}")

    instance = model_cls(**valid_params)
    return tool.call(instance)

async def cli_loop(doc_id: str):
    """Continuously prompt the user for input from the CLI."""
    global thread_id
    is_complete = True
    iterations = 0
    chat_result = None
    while True:
        if is_complete or iterations > MAX_ITERATIONS:
            iterations = 0
            # Create a task for user input and one for the comment event.
            input_task = asyncio.create_task(async_input("Enter your prompt (or 'exit' to quit): "))
            event_task = asyncio.create_task(new_comment_event.wait())

            # Wait until either input is received or a new comment is detected.
            done, pending = await asyncio.wait(
                [input_task, event_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # If a new comment was detected, cancel the input and jump to the next loop iteration.
            if event_task in done:
                new_comment_event.clear()  # Reset the event.
                input_task.cancel()         # Cancel the input task (it may still complete later, but we ignore it).
                print("New comment received; interrupting input and checking updates.")
                is_complete = False
                continue  # Jump back into the loop (or perform other actions if needed).

            # Otherwise, the user provided input.
            prompt = input_task.result()
            if prompt.strip().lower() == "exit":
                print("Exiting...")
                break

            chat_result = await chat(prompt)

        done_result = await check_done()
        is_complete = done_result['args'].is_complete

        iterations += 1
        if not is_complete and iterations < MAX_ITERATIONS:
            print(f"iterations {iterations}/{MAX_ITERATIONS}")
            chat_result = await chat(f"You're not quite done\n{done_result['args']}\nPlease continue")
        else:
            add_comment_result = await add_comment(doc_id=doc_id, comment=done_result['output'])
            iterations = 0

async def add_comment(doc_id: str, comment: str):
    create_comment_tool = tool_cache.get("create-comment")
    if not create_comment_tool:
        print("Tool 'create-comment' not found in cache.")
        return
    return call_tool_with_introspection(create_comment_tool, {"document_id": doc_id, "content": comment, "start_offset": 1, "length": 1 })





async def poll_comments(doc_id):
    read_comments_tool = tool_cache.get("read-comments")
    if not read_comments_tool:
        print("Tool 'read-comments' not found in cache.")
        return

    last_time = 0
    while True:
        try:
            result = call_tool_with_introspection(read_comments_tool, {"document_id": doc_id})
        except Exception as e:
            print("Error calling read-comments tool:", e)
            await asyncio.sleep(10)
            continue

        # Assume result['output'] is a string representation of a list of comment dicts.
        comments = ast.literal_eval(result['output'])
        for comment in comments:
            modified = datetime.strptime(comment['modifiedTime'][:-1], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            if modified > last_time:
                await chat(f"New comment detected in document {doc_id}, check comments")
                last_time = time.time()
                new_comment_event.set()  # Trigger the event.
        await asyncio.sleep(10)  # Adjust polling interval as needed

async def poll_doc_changes(doc_id):
    """Background task that polls for document changes using the 'read-doc' tool with debouncing."""
    global last_modified_time
    read_doc_tool = tool_cache.get("read-doc")
    if not read_doc_tool:
        print("Tool 'read-doc' not found in cache.")
        return

    last_result = None
    last_prompt_time = 0
    while True:
        try:
            result = call_tool_with_introspection(read_doc_tool, {"document_id": doc_id})
        except Exception as e:
            print("Error calling read-doc tool:", e)
            await asyncio.sleep(10)
            continue

        if last_result is None:
            last_result = result
        elif last_result != result:
            last_result = result
            current_time = time.time()
            if current_time - last_prompt_time > 30:
                print(f"Detected change in document {doc_id}, prompting model to check doc.")
                await chat(f"Doc changes detected for doc: {doc_id}")
                last_prompt_time = current_time
        await asyncio.sleep(10)  # Adjust polling interval as needed


async def create_doc():
    create_doc_tool = tool_cache.get("create-doc")
    result = call_tool_with_introspection(create_doc_tool, {"title": "New Doc"})
    print(f"Assistant response: {result["output"]}")
    doc_id = result["output"].split(":")[2].split("/")[5]
    return doc_id


async def main():
    global thread_id
    instructions = ("You are a long running agent that assists with google docs assist the user with docs related tasks."
                    "Additionally, you may be notified of doc changes and comments."
                    "For doc changes, first pull the latest version of the document and read it.  Then you can proceed to "
                    "edit it if it makes sense. "
                    "For comments, pull the latest comments and either reply to the comment or update the doc, or both"
                    "based on the comment.")
    init_manager(instructions)
    init_done_manager()
    #doc_id = await create_doc()
    doc_id = "1A4ZGfs-h2N145Blhmrj7vUFet5gNYTHnohs6O3JRsoE"
    await chat(prompt="What tools do you have access to? Make sure you use them going forward.")
    thread_id = assistant_manager.thread.id
    tasks = [
        asyncio.create_task(cli_loop(doc_id)),
        asyncio.create_task(poll_comments(doc_id)),
        asyncio.create_task(poll_doc_changes(doc_id))
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
