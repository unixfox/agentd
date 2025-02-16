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
from astra_assistants.mcp_openai_adapter import MCPRepresentationStdio, MCPRepresentation

DEFAULT_CONFIG = {
    "assistant_id": None,
    "thread_id": None,
    "done_assistant_id": None,
}


class ConfigManager:
    def __init__(self, app_name: str, app_author: str):
        self.app_name = app_name
        self.app_author = app_author
        self.config_dir = user_config_dir(app_name, app_author)
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        return self.config

    def save_config(self):
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()


class Agent:
    MAX_ITERATIONS = 5

    def __init__(self, instructions: str, config_manager: ConfigManager, mcps: MCPRepresentation):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.instructions = instructions

        # Managers for conversation and supervision.
        self.assistant_manager = None
        self.done_manager = None

        # State variables.
        self.thread_id = None
        self.assistant_id = None
        self.tool_cache = {}
        self.interaction_origin = "cli"
        self.current_comment_id = None
        self.new_comment_event = asyncio.Event()
        self.chat_result = None
        self.last_modified_time = None

        self._init_managers(mcps)

    def _init_managers(self, mcps: MCPRepresentation):
        # Initialize assistant_manager
        self.thread_id = self.config.get("thread_id")
        self.assistant_id = self.config.get("assistant_id")

        client = patch(OpenAI())
        self.assistant_manager = AssistantManager(
            client=client,
            model="gpt-4o",
            instructions=self.instructions,
            assistant_id=self.assistant_id,
            mcp_represenations=mcps,
        )
        if not self.assistant_id:
            self.config_manager.set("assistant_id", self.assistant_manager.assistant.id)
            self.assistant_id = self.assistant_manager.assistant.id

        self.cache_tools()

        # Initialize done_manager
        done_assistant_id = self.config.get("done_assistant_id")
        done_instructions = (
            "You are a supervisor agent, your job is to use the tool to determine if a task has "
            "been completed. For a task to be completed it must be reflected in the document itself, "
            "not just in the conversation. *NOTE* you will receive notifications when the document is updated "
            "(even by you). If you haven't received a notification, it means you haven't updated the doc. Use the edit_document tool!"
        )
        self.done_manager = AssistantManager(
            client=client,
            instructions=done_instructions,
            assistant_id=done_assistant_id,
            tools=[DoneTool()],
        )
        if not done_assistant_id:
            self.config_manager.set("done_assistant_id", self.done_manager.assistant.id)

    def cache_tools(self):
        """Caches available tools by name for fast lookup."""
        self.tool_cache = {}
        for tool in self.assistant_manager.tools:
            self.tool_cache[tool.mcp_tool.name] = tool

    @staticmethod
    def call_tool_with_introspection(tool, provided_params: dict):
        """
        Introspect the tool’s model to filter and validate parameters.
        """
        model_cls = tool.get_model()  # Assume a Pydantic-like model.
        valid_fields = set(model_cls.model_fields.keys())
        valid_params = {k: v for k, v in provided_params.items() if k in valid_fields}

        # Check for missing required fields.
        for field_name, field_info in model_cls.model_fields.items():
            if field_info.is_required() and field_name not in valid_params:
                raise ValueError(f"Missing required parameter: {field_name}")

        instance = model_cls(**valid_params)
        return tool.call(instance)

    async def async_input(self, prompt: str) -> str:
        """Async wrapper for blocking input."""
        return await asyncio.to_thread(input, prompt)

    async def check_done(self):
        """Checks with the supervisor agent if the current task is complete."""
        tool = DoneTool()
        result = await self.done_manager.run_thread(
            content="Has the last user request been completed?",
            thread_id=self.thread_id,
            tool=tool,
        )
        return result

    async def chat(self, prompt: str):
        """Send a prompt to the assistant manager and display the response."""
        self.chat_result = await self.assistant_manager.run_thread(content=prompt, thread_id=self.thread_id)
        arguments = self.chat_result.get("arguments")
        if arguments is not None:
            print(
                f"Tool Call: {arguments.__class__.__name__}({arguments}) \nTool Call Result: {self.chat_result['output']}"
            )
        print("Assistant response:")
        print(self.chat_result.get("text", "No text response found."))
        return self.chat_result

    async def add_comment(self, doc_id: str, comment: str):
        """Add a new comment using the appropriate tool."""
        create_comment_tool = self.tool_cache.get("create-comment")
        if not create_comment_tool:
            print("Tool 'create-comment' not found in cache.")
            return
        return self.call_tool_with_introspection(create_comment_tool, {"document_id": doc_id, "content": comment})

    async def reply_comment(self, doc_id: str, comment_id: str, reply: str):
        """
        Reply to a comment. If a dedicated reply-comment tool isn’t available,
        it will fall back to creating a normal comment.
        """
        reply_comment_tool = self.tool_cache.get("reply-comment")
        if not reply_comment_tool:
            raise ValueError("reply-comment tool is missing")
        return self.call_tool_with_introspection(
            reply_comment_tool,
            {"document_id": doc_id, "comment_id": comment_id, "reply": f"[BOT COMMENT]:\n{reply}"},
        )

    async def cli_loop(self, doc_id: str):
        """CLI loop to interact with the agent, either from the terminal or a comment."""
        is_complete = True
        iterations = 0
        self.chat_result = None

        while True:
            if is_complete or iterations > self.MAX_ITERATIONS:
                iterations = 0
                input_task = asyncio.create_task(self.async_input("Enter your prompt (or 'exit' to quit): "))
                event_task = asyncio.create_task(self.new_comment_event.wait())

                done, _ = await asyncio.wait([input_task, event_task], return_when=asyncio.FIRST_COMPLETED)

                # A comment event was received.
                if event_task in done:
                    self.new_comment_event.clear()
                    input_task.cancel()
                    print("New comment received; switching to comment interaction.")
                    self.interaction_origin = "comment"
                    is_complete = False
                    continue

                # CLI input was received.
                prompt = input_task.result()
                if prompt.strip().lower() == "exit":
                    print("Exiting...")
                    break
                await self.chat(prompt)

            print(self.chat_result)

            # If the response calls a tool, let it finish before counting an iteration.
            if "arguments" in self.chat_result:
                await self.chat("continue")
                continue

            done_result = await self.check_done()
            is_complete = done_result["args"].is_complete
            iterations += 1

            if not is_complete and iterations < self.MAX_ITERATIONS:
                print(f"iterations {iterations}/{self.MAX_ITERATIONS}")
                await self.chat(f"You're not quite done:\n{done_result['args']}\nPlease continue")
            elif is_complete:
                summary_result = await self.chat("Looks like you completed the task, please summarize your work.")
                if self.interaction_origin == "comment":
                    await self.reply_comment(
                        doc_id=doc_id,
                        comment_id=self.current_comment_id,
                        reply=f"Task Completed: {summary_result.get('text', 'No summary provided.')}",
                    )
                    self.interaction_origin = "cli"
                    self.current_comment_id = None
            else:
                summary_result = await self.chat("It sounds like you're having difficulty with this task")
                if self.interaction_origin == "comment":
                    reply_result = await self.reply_comment(
                        doc_id=doc_id,
                        comment_id=self.current_comment_id,
                        reply=summary_result.get("text", "No summary provided."),
                    )
                    print("Posted reply comment:", reply_result.get("output", "No output"))
                    self.interaction_origin = "cli"
                    self.current_comment_id = None

    async def poll_comments(self, doc_id: str):
        """Poll for new comments in the document and notify the agent."""
        read_comments_tool = self.tool_cache.get("read-comments")
        if not read_comments_tool:
            print("Tool 'read-comments' not found in cache.")
            return

        last_time = 0
        while True:
            try:
                result = self.call_tool_with_introspection(read_comments_tool, {"document_id": doc_id})
            except Exception as e:
                print("Error calling read-comments tool:", e)
                await asyncio.sleep(10)
                continue

            comments = ast.literal_eval(result["output"])
            for comment in comments:
                modifiedTime = datetime.strptime(comment["modifiedTime"][:-1], "%Y-%m-%dT%H:%M:%S.%f").replace(
                    tzinfo=timezone.utc
                ).timestamp()
                if 'resolved' in comment and comment['resolved']:
                    continue
                if modifiedTime > last_time:
                    # Skip comments already replied to.
                    if "replies" in comment and comment["replies"] and comment["replies"][-1]["content"].startswith(
                            "[BOT COMMENT]"
                    ):
                        continue
                    self.interaction_origin = "comment"
                    self.current_comment_id = comment.get("id")
                    await self.reply_comment(doc_id, comment["id"], "👀")
                    print(f"New comment detected in document {doc_id}.")
                    await self.chat(f"New comment detected in document {doc_id}: {comment}.")
                    last_time = time.time()
                    self.new_comment_event.set()
            await asyncio.sleep(10)

    async def poll_doc_changes(self, doc_id: str):
        """Poll for document changes and notify the agent if the document was updated."""
        read_doc_tool = self.tool_cache.get("read-doc")
        if not read_doc_tool:
            print("Tool 'read-doc' not found in cache.")
            return

        last_result = None
        last_prompt_time = 0
        while True:
            try:
                result = self.call_tool_with_introspection(read_doc_tool, {"document_id": doc_id})
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
                    print(f"Detected change in document {doc_id}.")
                    await self.chat(f"Doc changes detected for doc: {doc_id}:\n{result}")
                    last_prompt_time = current_time
            await asyncio.sleep(10)

    async def create_doc(self):
        """Create a new document using the create-doc tool."""
        create_doc_tool = self.tool_cache.get("create-doc")
        result = self.call_tool_with_introspection(create_doc_tool, {"title": "New Doc"})
        print(f"Assistant response: {result['output']}")
        doc_id = result["output"].split(":")[2].split("/")[5]
        return doc_id

    async def clear_acks(self, doc_id: str):
        """Clear acknowledgment replies in the document."""
        read_comments_tool = self.tool_cache.get("read-comments")
        if not read_comments_tool:
            print("Tool 'read-comments' not found in cache.")
            return
        delete_reply_tool = self.tool_cache.get("delete-reply")
        if not delete_reply_tool:
            print("Tool 'delete-reply' not found in cache.")
            return

        try:
            result = self.call_tool_with_introspection(read_comments_tool, {"document_id": doc_id})
        except Exception as e:
            print("Error calling read-comments tool:", e)
            return

        comments = ast.literal_eval(result["output"])
        for comment in comments:
            if "replies" in comment and comment["replies"]:
                last_reply = comment["replies"][-1]
                if last_reply["content"] == "[BOT COMMENT]:\n👀":
                    comment_id = comment["id"]
                    reply_id = last_reply["id"]
                    try:
                        self.call_tool_with_introspection(
                            delete_reply_tool,
                            {"document_id": doc_id, "comment_id": comment_id, "reply_id": reply_id},
                        )
                    except Exception as e:
                        print("Error calling delete-reply tool:", e)
                        return

    async def run(self, doc_id: str):
        """Start the agent’s main event loop."""
        await self.clear_acks(doc_id)
        await self.chat(prompt="What tools do you have access to? Make sure you use them going forward.")
        self.thread_id = self.assistant_manager.thread.id
        tasks = [
            asyncio.create_task(self.cli_loop(doc_id)),
            asyncio.create_task(self.poll_comments(doc_id)),
            asyncio.create_task(self.poll_doc_changes(doc_id)),
        ]
        await asyncio.gather(*tasks)


async def main():
    instructions = (
        "You are a long running agent that assists with google docs and helps the user with docs related tasks. "
        "Additionally, you may be notified of doc changes and comments. "
        "When in doubt, ensure you are looking at the latest version of the doc. "
        "Don't use any markdown in the google doc."
    )
    config_manager = ConfigManager(app_name="agentd", app_author="phact")

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
                "../mcp-google-docs/.auth/token",
            ],
            tool_filter=['read-doc','rewrite-document']
        )
    ]

    agent = Agent(instructions, config_manager, mcps)
    # Uncomment the next line to create a new document
    # doc_id = await agent.create_doc()
    doc_id = "1A4ZGfs-h2N145Blhmrj7vUFet5gNYTHnohs6O3JRsoE"
    await agent.run(doc_id)


if __name__ == "__main__":
    asyncio.run(main())
