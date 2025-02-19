import ast
import json
import os
import asyncio
import time
import re
from loguru import logger
from datetime import datetime, timezone

from config import ConfigManager
from agentd.tools.done_tool import DoneTool
from openai import OpenAI
from astra_assistants import patch
from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.mcp_openai_adapter import MCPRepresentationStdio, MCPRepresentation

# Import our subscription classes.
from subscriptions import GoogleDocSubscription, SlackSubscription

class Agentd:
    MAX_ITERATIONS = 5

    def __init__(self, instructions: str, config_manager: ConfigManager, mcps: MCPRepresentation):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.instructions = instructions

        self.assistant_manager: AssistantManager
        self.done_manager = None

        self.thread_id = None
        self.assistant_id = None
        self.tool_cache = {}
        self.interaction_origin = "cli"
        self.current_comment_id = None
        self.new_comment_event = asyncio.Event()
        self.chat_result = None
        self.last_modified_time = None

        self.exit_event = asyncio.Event()
        self.doc_id = None

        self._init_managers(mcps)

    def _init_managers(self, mcps: MCPRepresentation):
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
            logger.info(f"New assistant created with id: {self.assistant_id}")

        self.cache_tools()

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
            logger.info(f"New done assistant created with id: {self.done_manager.assistant.id}")

    def cache_tools(self):
        self.tool_cache = {}
        for tool in self.assistant_manager.tools:
            self.tool_cache[tool.mcp_tool.name] = tool
        logger.debug(f"Cached tools: {list(self.tool_cache.keys())}")

    @staticmethod
    def call_tool_with_introspection(tool, provided_params: dict):
        model_cls = tool.get_model()
        valid_fields = set(model_cls.model_fields.keys())
        valid_params = {k: v for k, v in provided_params.items() if k in valid_fields}

        for field_name, field_info in model_cls.model_fields.items():
            if field_info.is_required() and field_name not in valid_params:
                raise ValueError(f"Missing required parameter: {field_name}")

        instance = model_cls(**valid_params)
        return tool.call(instance)

    async def async_input(self, prompt: str) -> str:
        return await asyncio.to_thread(input, prompt)

    async def check_done(self):
        tool = DoneTool()
        result = await self.done_manager.run_thread(
            content="Has the last user request been completed?",
            thread_id=self.thread_id,
            tool=tool,
        )
        logger.debug(f"Done check result: {result}")
        return result

    def extract_rewrite_content(self, text: str) -> str:
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content = matches[0].strip()
            logger.debug(f"Extracted rewrite content: {content}")
            return content
        return None

    def process_rewrite_content(self, content: str):
        rewrite_tool = self.tool_cache.get("rewrite-document")
        if not rewrite_tool:
            logger.error("Tool 'rewrite-document' not found in cache.")
            return
        try:
            result = self.call_tool_with_introspection(
                rewrite_tool, {"document_id": self.doc_id, "final_text": content}
            )
            logger.info(f"Rewrite-document tool call result: {result.get('output', 'No output')}")
            return result
        except Exception as e:
            logger.error(f"Error calling rewrite-document tool: {e}")
            return None

    async def chat(self, prompt: str):
        self.chat_result = await self.assistant_manager.run_thread(content=prompt, thread_id=self.thread_id)
        arguments = self.chat_result.get("arguments")
        if arguments is not None:
            logger.info(f"Tool Call: {arguments.__class__.__name__}({arguments}) \nTool Call Result: {self.chat_result['output']}")
            # If the LLM requested a create-doc tool call, trigger new document creation.
            if self.chat_result['arguments'].__class__.__name__ == 'CreateDoc':
                new_doc_id = self.chat_result["text"].split(":")[2].split("/")[5]
                logger.info(f"LLM requested new document creation: {new_doc_id}")
                # Spawn a new agent instance for the new document.
                asyncio.create_task(run_new_agent_instance(new_doc_id))
        text = self.chat_result.get("text", "No text response found.")
        logger.info(f"Assistant response: {text}")

        rewrite_content = self.extract_rewrite_content(text)
        if rewrite_content and self.doc_id:
            logger.info("Detected rewrite content in triple backticks; calling rewrite-document tool...")
            self.process_rewrite_content(rewrite_content)
            await self.chat(f"Here is the current content of the document: {rewrite_content}")
        return self.chat_result

    async def add_comment(self, doc_id: str, comment: str):
        create_comment_tool = self.tool_cache.get("create-comment")
        if not create_comment_tool:
            logger.error("Tool 'create-comment' not found in cache.")
            return
        return self.call_tool_with_introspection(create_comment_tool, {"document_id": doc_id, "content": comment})

    async def reply_comment(self, doc_id: str, comment_id: str, reply: str):
        reply_comment_tool = self.tool_cache.get("reply-comment")
        if not reply_comment_tool:
            raise ValueError("reply-comment tool is missing")
        return self.call_tool_with_introspection(
            reply_comment_tool,
            {"document_id": doc_id, "comment_id": comment_id, "reply": f"[BOT COMMENT]:\n{reply}"},
        )

    async def cli_loop(self, doc_id: str):
        is_complete = True
        iterations = 0
        self.chat_result = None

        while True:
            if is_complete or iterations > self.MAX_ITERATIONS:
                iterations = 0
                input_task = asyncio.create_task(self.async_input("Enter your prompt (or 'exit' to quit): "))
                event_task = asyncio.create_task(self.new_comment_event.wait())

                done, _ = await asyncio.wait([input_task, event_task], return_when=asyncio.FIRST_COMPLETED)

                if event_task in done:
                    self.new_comment_event.clear()
                    input_task.cancel()
                    logger.info("New comment received; switching to comment interaction.")
                    self.interaction_origin = "comment"
                    is_complete = False
                    continue

                prompt = input_task.result()
                if prompt.strip().lower() == "exit":
                    logger.info("Exiting...")
                    self.exit_event.set()
                    break
                await self.chat(prompt)

            logger.debug(f"Chat result: {self.chat_result}")

            if "arguments" in self.chat_result:
                await self.chat("continue")
                continue

            await self.chat("continue")
            done_result = await self.check_done()
            is_complete = done_result["args"].is_complete
            iterations += 1

            if not is_complete and iterations < self.MAX_ITERATIONS:
                logger.info(f"iterations {iterations}/{self.MAX_ITERATIONS}")
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
                    logger.info(f"Posted reply comment: {reply_result.get('output', 'No output')}")
                    self.interaction_origin = "cli"
                    self.current_comment_id = None

    async def clear_acks(self, doc_id: str):
        read_comments_tool = self.tool_cache.get("read-comments")
        if not read_comments_tool:
            logger.error("Tool 'read-comments' not found in cache.")
            return
        delete_reply_tool = self.tool_cache.get("delete-reply")
        if not delete_reply_tool:
            logger.error("Tool 'delete-reply' not found in cache.")
            return

        try:
            result = self.call_tool_with_introspection(read_comments_tool, {"document_id": doc_id})
        except Exception as e:
            logger.error(f"Error calling read-comments tool: {e}")
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
                        logger.debug(f"Cleared ack for comment {comment_id}")
                    except Exception as e:
                        logger.error(f"Error calling delete-reply tool: {e}")
                        return

    async def run(self, doc_id: str):
        self.doc_id = doc_id
        await self.clear_acks(doc_id)
        await self.chat(prompt="What tools do you have access to? Make sure you use them going forward.")
        self.thread_id = self.assistant_manager.thread.id

        # Create modular subscriptions.
        subscriptions = []
        google_sub = GoogleDocSubscription(self, doc_id)
        subscriptions.append(google_sub)
        if self.config.get("enable_slack_polling", True):
            list_channels_tool = self.tool_cache.get("slack_list_channels")
            if not list_channels_tool:
                logger.error("Tool 'slack_list_channels' not found in cache.")
                return
            result = self.call_tool_with_introspection(list_channels_tool, {"limit": 10})
            output = json.loads(result["output"])
            channels = [channel for channel in output.get('channels', [])
                        if channel.get('name') == 'all-jake-test-ground']
            if not channels:
                logger.error("No channel found with name 'all-jake-test-ground'.")
                return
            channel_id = channels[0]['id']
            slack_sub = SlackSubscription(self, channel_id)
            subscriptions.append(slack_sub)

        tasks = [
            asyncio.create_task(self.cli_loop(doc_id)),
            *[asyncio.create_task(sub.poll()) for sub in subscriptions],
        ]
        await asyncio.gather(*tasks)

    async def create_doc(self):
        create_doc_tool = self.tool_cache.get("create-doc")
        result = self.call_tool_with_introspection(create_doc_tool, {"title": "New Doc"})
        logger.info(f"Assistant response: {result['output']}")
        doc_id = result["output"].split(":")[2].split("/")[5]
        return doc_id

# Helper to run a new agent instance using a given doc_id.
async def run_new_agent_instance(new_doc_id: str):
    instructions = (
        "You are a long running document writing agent. "
        "You will be notified of doc changes and comments. "
        "When you want to update the document, please enclose your new content in triple backticks (```your content```). "
        "I will then automatically update the document using that content. "
        "When in doubt, ensure you are looking at the latest version of the doc. "
        "Don't use any markdown in the google doc. "
        "Always include URLs in your citations when referencing content from search tools."
    )
    config_manager = ConfigManager(app_name="agentd", app_author="phact")
    brave_key = os.environ['BRAVE_API_KEY']
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
            tool_filter=['read-doc', 'create-doc']
        ),
        MCPRepresentationStdio(
            type="stdio",
            command="npx",
            arguments=["-y", "@modelcontextprotocol/server-brave-search"],
            env_vars=[f"BRAVE_API_KEY={brave_key}"]
        ),
        MCPRepresentationStdio(
            type="stdio",
            command="uv",
            arguments=["run", "--refresh", "--with", "mcp_simple_arxiv", "mcp-simple-arxiv"],
        ),
        MCPRepresentationStdio(
            type="stdio",
            command="npx",
            arguments=["-y", "@modelcontextprotocol/server-slack"],
            env_vars=[
                f"SLACK_BOT_TOKEN={os.environ['SLACK_BOT_TOKEN']}",
                f"SLACK_TEAM_ID={os.environ['SLACK_TEAM_ID']}",
            ]
        )
    ]
    new_agent = Agentd(instructions, config_manager, mcps)
    logger.info(f"Starting new agent instance with doc: {new_doc_id}")
    await new_agent.run(new_doc_id)

# This function creates an initial agent instance by having the agent create its own document.
async def run_agent_instance():
    instructions = (
        "You are a long running document writing agent. "
        "You will be notified of doc changes and comments. "
        "When you want to update the document, please enclose your new content in triple backticks (```your content```). "
        "I will then automatically update the document using that content. "
        "When in doubt, ensure you are looking at the latest version of the doc. "
        "Don't use any markdown in the google doc. "
        "Always include URLs in your citations when referencing content from search tools."
        "When you create documents make sure to share them with datastax.com. "
    )
    config_manager = ConfigManager(app_name="agentd", app_author="phact")
    brave_key = os.environ['BRAVE_API_KEY']
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
            tool_filter=['read-doc', 'create-doc']
        ),
        MCPRepresentationStdio(
            type="stdio",
            command="npx",
            arguments=["-y", "@modelcontextprotocol/server-brave-search"],
            env_vars=[f"BRAVE_API_KEY={brave_key}"]
        ),
        MCPRepresentationStdio(
            type="stdio",
            command="uv",
            arguments=["run", "--refresh", "--with", "mcp_simple_arxiv", "mcp-simple-arxiv"],
        ),
        MCPRepresentationStdio(
            type="stdio",
            command="npx",
            arguments=["-y", "@modelcontextprotocol/server-slack"],
            env_vars=[
                f"SLACK_BOT_TOKEN={os.environ['SLACK_BOT_TOKEN']}",
                f"SLACK_TEAM_ID={os.environ['SLACK_TEAM_ID']}",
            ]
        )
    ]
    agent = Agentd(instructions, config_manager, mcps)
    doc_id = await agent.create_doc()
    logger.info(f"New document created: {doc_id}")
    await agent.run(doc_id)

async def main():
    await run_agent_instance()

if __name__ == "__main__":
    asyncio.run(main())
