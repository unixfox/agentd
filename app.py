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
assistant_manager: AssistantManager = None
done_manager: AssistantManager = None
thread_id = None
assistant_id = None
chat_result = None

# Global caches to detect changes
last_modified_time = None

# Global tool cache
tool_cache = {}

# New global variables to track interaction origin:
# - interaction_origin: "cli" or "comment"
# - current_comment_id: if a comment triggered the interaction, store its id here.
interaction_origin = "cli"
current_comment_id = None

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
    global thread_id, chat_result
    chat_result = await assistant_manager.run_thread(content=prompt, thread_id=thread_id)
    arguments = chat_result.get("arguments")
    if arguments is not None:
        print(f"Tool Call: {arguments.__class__.__name__}({arguments}) \nTool Call Result: {chat_result['output']}")

    print("Assistant response:")
    print(chat_result.get('text', 'No text response found.'))
    return chat_result

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

    instructions = (
        "You are a supervisor agent, your job is to use the tool to determine if a task has "
        "been completed. For a task to be completed it has reflect in the document itself not just the conversation."
        "*NOTE* you will receive notifications when the document is updated (even by you)."
        "If you haven't received a notification, it means you haven't updated the doc. Use the edit_document tool!"
    )

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

async def add_comment(doc_id: str, comment: str):
    create_comment_tool = tool_cache.get("create-comment")
    if not create_comment_tool:
        print("Tool 'create-comment' not found in cache.")
        return
    return call_tool_with_introspection(create_comment_tool, {
        "document_id": doc_id,
        "content": comment,
    })

async def reply_comment(doc_id: str, comment_id: str, reply: str):
    """
    Helper function to post a reply comment.
    If a dedicated reply-comment tool isn’t available, we fall back to create a normal comment.
    """
    reply_comment_tool = tool_cache.get("reply-comment")
    assert reply_comment_tool, "reply-comment tool is missing"
    return call_tool_with_introspection(reply_comment_tool, {
        "document_id": doc_id,
        "comment_id": comment_id,
        "reply": f"[BOT COMMENT]:\n{reply}",
    })

async def cli_loop(doc_id: str):
    """Continuously prompt the user for input from the CLI.
       This loop now distinguishes between CLI-initiated interactions and those started via a comment.
    """
    global thread_id, interaction_origin, current_comment_id, chat_result
    is_complete = True
    iterations = 0
    chat_result = None

    while True:
        if is_complete or iterations > MAX_ITERATIONS:
            iterations = 0
            # Create tasks for user input and waiting for a new comment event.
            input_task = asyncio.create_task(async_input("Enter your prompt (or 'exit' to quit): "))
            event_task = asyncio.create_task(new_comment_event.wait())

            done, pending = await asyncio.wait(
                [input_task, event_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # If a comment event occurred, we treat this as a comment interaction.
            if event_task in done:
                new_comment_event.clear()  # Reset the event.
                input_task.cancel()         # Cancel the input task.
                print("New comment received; switching to comment interaction.")
                interaction_origin = "comment"
                # current_comment_id is set in poll_comments.
                is_complete = False
                continue  # Restart loop to begin processing the comment interaction.

            # Otherwise, we got user input from the CLI.
            prompt = input_task.result()
            if prompt.strip().lower() == "exit":
                print("Exiting...")
                break
            await chat(prompt)

        print(chat_result)

        # tool calls don't count as iterations
        if "arguments" in chat_result:
            await chat("continue")
            continue

        # Check if the current interaction is complete.
        done_result = await check_done()
        is_complete = done_result['args'].is_complete
        iterations += 1

        if not is_complete and iterations < MAX_ITERATIONS:
            print(f"iterations {iterations}/{MAX_ITERATIONS}")
            await chat(f"You're not quite done:\n{done_result['args']}\nPlease continue")
        elif is_complete:
            summary_result = await chat("Looks like you completed the task, please summarize your work.")
            if interaction_origin == "comment":
                reply_result = await reply_comment(
                    doc_id=doc_id,
                    comment_id=current_comment_id,
                    reply=f"Task Completed: {summary_result.get('text', 'No summary provided.')}",
                )
                # Reset to CLI mode for the next interaction.
                interaction_origin = "cli"
                current_comment_id = None
        else:
            # We've exceeded the iteration limit.
            summary_result = await chat("It sounds like you're having difficulty with this task")
            if interaction_origin == "comment":
                reply_result = await reply_comment(
                    doc_id=doc_id,
                    comment_id=current_comment_id,
                    reply=summary_result.get('text', 'No summary provided.'),
                )
                print("Posted reply comment:", reply_result.get('output', 'No output'))
                # Reset to CLI mode for the next interaction.
                interaction_origin = "cli"
                current_comment_id = None

async def poll_comments(doc_id):
    """Polls for new comments. If a new comment is detected, sets the interaction mode to 'comment'
       and stores the comment ID for later reply.
    """
    read_comments_tool = tool_cache.get("read-comments")
    if not read_comments_tool:
        print("Tool 'read-comments' not found in cache.")
        return

    global interaction_origin, current_comment_id
    last_time = 0
    while True:
        try:
            result = call_tool_with_introspection(read_comments_tool, {"document_id": doc_id})
        except Exception as e:
            print("Error calling read-comments tool:", e)
            await asyncio.sleep(10)
            continue

        comments = ast.literal_eval(result['output'])
        for comment in comments:
            modifiedTime = datetime.strptime(comment['modifiedTime'][:-1], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            if modifiedTime > last_time:
                if 'replies' in comment and len(comment['replies']) > 0 and comment['replies'][len(comment['replies'])-1]['content'].startswith("[BOT COMMENT]"):
                    continue
                # Set interaction as comment-based and store the comment ID.
                interaction_origin = "comment"
                current_comment_id = comment.get("id")

                # ack the comment
                await reply_comment(doc_id, comment['id'],"👀")
                await chat(f"New comment detected in document {doc_id}, please check comments.")

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
    print(f"Assistant response: {result['output']}")
    doc_id = result["output"].split(":")[2].split("/")[5]
    return doc_id


async def clear_acks(doc_id):
    read_comments_tool = tool_cache.get("read-comments")
    if not read_comments_tool:
        print("Tool 'read-comments' not found in cache.")
        return
    delete_reply_tool = tool_cache.get("delete-reply")
    if not read_comments_tool:
        print("Tool 'read-comments' not found in cache.")
        return

    try:
        result = call_tool_with_introspection(read_comments_tool, {"document_id": doc_id})
    except Exception as e:
        print("Error calling read-comments tool:", e)
        return

    comments = ast.literal_eval(result['output'])
    for comment in comments:
        if 'replies' in comment and len(comment['replies']) > 0:
            last_reply = comment['replies'][len(comment['replies'])-1]
            last_reply_content = last_reply['content']
            if last_reply_content == "[BOT COMMENT]:\n👀":
                comment_id = comment['id']
                reply_id = last_reply['id']
                try:
                    call_tool_with_introspection(delete_reply_tool, {
                        "document_id": doc_id,
                        "comment_id": comment_id,
                        "reply_id": reply_id
                    })
                except Exception as e:
                    print("Error calling delete-reply tool:", e)
                    return

async def main():
    global thread_id
    instructions = (
        "You are a long running agent that assists with google docs and helps the user with docs related tasks. "
        "Additionally, you may be notified of doc changes and comments."
        "When in doubt, ensure you are looking at the latest version of the doc."
        "Don't use any markdown in the google doc."
    )
    init_manager(instructions)
    init_done_manager()
    #doc_id = await create_doc()
    doc_id = "1A4ZGfs-h2N145Blhmrj7vUFet5gNYTHnohs6O3JRsoE"
    await clear_acks(doc_id)
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
