import ast
import json
import os
import asyncio
import re
from loguru import logger

from agentd.section_chunker import wrap_sections, unwrap_sections
from agentd.config import ConfigManager
from agentd.tool_util import call_tool_with_introspection
from agentd.tools.done_tool import DoneTool
from openai import OpenAI
from astra_assistants import patch
from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.mcp_openai_adapter import MCPRepresentationStdio, MCPRepresentation

# Import our subscription classes.
from agentd.subscriptions import GoogleDocSubscription, SlackSubscription


def extract_rewrite_content(text: str):
    """
    Extracts rewrite content from the LLM response.
    Expects one or more code blocks in the following format:
      ```[target]
      <updated code>
      ```
    If [target] is provided (e.g. "file1.section1"), that section is updated;
    if empty, the entire file should be replaced.

    Returns:
      A list of code block strings if one or more are found; otherwise, None.
    """
    pattern = r"```(\S*)\n(.*?)\n```"
    blocks = ""
    for match in re.finditer(pattern, text, re.DOTALL):
        target = match.group(1).strip()
        updated_code = match.group(2).strip()
        logger.debug(f"Extracted rewrite content. Target: '{target}', Code: {updated_code[:60]}...")
        if blocks != "":
            blocks = f"{blocks}\n```{target}\n{updated_code}\n```"
        else:
            blocks = f"```{target}\n{updated_code}\n```"
    return blocks


class Agentd:
    MAX_ITERATIONS = 5

    def __init__(self, instructions: str, config_manager: ConfigManager, mcps: MCPRepresentation, doc_id: str = None, channel_id: str = None, assistant_id: str = None, done_assistant_id: str = None):
        self.thread_id = None
        self.assistant_manager: AssistantManager = None
        self.done_manager = None
        self.current_comment_id = None
        self.chat_result = None
        self.last_modified_time = None
        self.doc_state = None

        self.doc_id = doc_id
        self.channel_id = channel_id
        self.config_manager = config_manager
        self.config = config_manager.config
        self.instructions = instructions
        self.assistant_id = assistant_id
        self.done_assistant_id = done_assistant_id
        self.tool_cache = {}
        self.interaction_origin = "cli"
        self.new_comment_event = asyncio.Event()
        self.exit_event = asyncio.Event()
        self._init_managers(mcps)
        if self.doc_id:
            self.load_document_state()

    def update_agent_config_field(self, field, value):
        # Determine agent type and key
        agent_type = "channels" if self.channel_id else "documents"
        agent_key = self.channel_id if self.channel_id else self.doc_id

        # Ensure the nested dictionaries exist
        agents_config = self.config_manager.config.setdefault("agents", {})
        agent_group = agents_config.setdefault(agent_type, {})
        agent = agent_group.setdefault(agent_key, {})

        # Update the field
        agent[field] = value

        # Save the updated config
        self.config_manager.save_config()

    def _init_managers(self, mcps: MCPRepresentation):
        client = patch(OpenAI())
        self.assistant_manager = AssistantManager(
            client=client,
            #model="gpt-4o",
            model="o3-mini",
            instructions=self.instructions,
            assistant_id=self.assistant_id,
            mcp_represenations=mcps,
            thread_id=self.thread_id
        )
        if not self.assistant_id:
            self.assistant_id = self.assistant_manager.assistant.id
            self.update_agent_config_field("assistant_id", self.assistant_id)
            logger.info(f"New assistant created with id: {self.assistant_id}")

        # If thread_id is not set, use the assistant manager's thread id and save it.
        if not self.thread_id:
            self.thread_id = self.assistant_manager.thread.id
            self.update_agent_config_field("thread_id", self.thread_id)
            logger.info(f"Thread id set to: {self.thread_id}")

        self.cache_tools()

        done_assistant_id = self.done_assistant_id
        done_instructions = (
            "You are a supervisor agent, your job is to use the tool to determine if a task has "
            "been completed. For a task to be completed it must be reflected in the document itself, "
            "not just in the conversation."
            "When editing existing documents make sure the edit was applied properly based on the user's intent"
            "(including appending instead of only replacing)."
            "Be pedantic about formatting and ensure only intended text makes it into the document."
        )
        self.done_manager = AssistantManager(
            client=client,
            instructions=done_instructions,
            assistant_id=done_assistant_id,
            tools=[DoneTool()],
        )
        if not self.done_assistant_id:
            self.done_assistant_id = self.done_manager.assistant.id
            self.update_agent_config_field("done_assistant_id", self.done_assistant_id)
            logger.info(f"New done assistant created with id: {self.done_assistant_id}")

    def cache_tools(self):
        self.tool_cache = {}
        for tool in self.assistant_manager.tools:
            self.tool_cache[tool.mcp_tool.name] = tool
        logger.debug(f"Cached tools: {list(self.tool_cache.keys())}")

    async def async_input(self, prompt: str) -> str:
        return await asyncio.to_thread(input, prompt)

    async def check_done(self):
        tool = DoneTool()
        result = await self.done_manager.run_thread(
            content=(
                f"Has the last user request been completed?\n"
                f"The current as displayed to the user is: \n{unwrap_sections(self.doc_state)}\n"
                f"The current with sections for editing is: \n{self.doc_state}\n"
            ),
            thread_id=self.thread_id,
            tool=tool,
        )
        logger.debug(f"Done check result: {result}")
        return result

    def load_document_state(self):
        """Fetch the current document content, split it into sections, and store it in internal state."""
        read_doc_tool = self.tool_cache.get("read-doc")
        if read_doc_tool:
            result = call_tool_with_introspection(read_doc_tool, {"document_id": self.doc_id})
            raw_content = result.get("output", "")

            if raw_content:
                self.doc_state = wrap_sections(raw_content)  # Convert to sectioned format
                logger.info("Document state loaded and sectioned successfully.")
            else:
                logger.warning("Empty document retrieved; no sections created.")
        else:
            logger.error("Tool 'read-doc' not found in cache.")

    def process_rewrite_content(self, content: str):
        """
        Applies the rewrite content using the section-based strategy.
        Update the internal state and call the rewrite-document tool with the new content.
        """
        rewrite_tool = self.tool_cache.get("rewrite-document")
        if not rewrite_tool:
            logger.error("Tool 'rewrite-document' not found in cache.")
            return

        from agentd.editors.section_based_editor import SectionBasedEditor
        if not self.doc_state:
            logger.warning("Internal document state not initialized; loading state.")
            # Optionally, load state synchronously or asynchronously here.
            self.load_document_state()

        try:
            coder = SectionBasedEditor()
            # Update internal state based on the edit.
            updated_content = coder.apply_edits(self.doc_state, content)
            updated_content = unwrap_sections(updated_content)
            call_tool_with_introspection(
                rewrite_tool, {"document_id": self.doc_id, "final_text": updated_content}
            )
            self.load_document_state()
            return self.doc_state
        except Exception as e:
            logger.error(f"Error applying section-based update: {e}")
            return None

    async def chat(self, prompt: str):
        self.chat_result = await self.assistant_manager.run_thread(content=prompt, thread_id=self.thread_id)
        arguments = self.chat_result.get("arguments")
        if arguments is not None:
            logger.info(
                f"Tool Call: {arguments.__class__.__name__}({arguments}) \nTool Call Result: {self.chat_result['output']}")
            # If the LLM requested a create-doc tool call, trigger new document creation.
            if self.chat_result['arguments'].__class__.__name__ == 'CreateDoc':
                chat_text = self.chat_result["text"]
                if "https://docs.google.com/document/d" in chat_text and len(chat_text.split("https:")) >= 2:
                    new_doc_id = self.chat_result["text"].split("https:")[1].split("/")[5]
                    logger.info(f"LLM requested new document creation: {new_doc_id}")
                    # Spawn a new agent instance for the new document.
                    asyncio.create_task(run_agent_instance(doc_id=new_doc_id, mcps=load_mcps_for("documents")))
        text = self.chat_result.get("text", "No text response found.")
        logger.info(f"Assistant response: {text}")

        rewrite_content = extract_rewrite_content(text)
        if rewrite_content and self.doc_id:
            logger.info("Detected rewrite content; applying section-based update...")
            rewrite_result = self.process_rewrite_content(rewrite_content)
            # await self.chat(f"Here is the updated document content:\n{rewrite_result}")
        return self.chat_result

    async def add_comment(self, doc_id: str, comment: str):
        create_comment_tool = self.tool_cache.get("create-comment")
        if not create_comment_tool:
            logger.error("Tool 'create-comment' not found in cache.")
            return
        return call_tool_with_introspection(create_comment_tool, {"document_id": doc_id, "content": comment})

    async def reply_comment(self, doc_id: str, comment_id: str, reply: str):
        reply_comment_tool = self.tool_cache.get("reply-comment")
        if not reply_comment_tool:
            raise ValueError("reply-comment tool is missing")
        return call_tool_with_introspection(
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

            # if "arguments" in self.chat_result:
            #    await self.chat("continue, ensure you apply the edit(s) with backticks and carefully number the sections")
            #    continue

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
            result = call_tool_with_introspection(read_comments_tool, {"document_id": doc_id})
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
                        call_tool_with_introspection(
                            delete_reply_tool,
                            {"document_id": doc_id, "comment_id": comment_id, "reply_id": reply_id},
                        )
                        logger.debug(f"Cleared ack for comment {comment_id}")
                    except Exception as e:
                        logger.error(f"Error calling delete-reply tool: {e}")
                        return

    async def run(self):
        if self.doc_id:
            await self.clear_acks(self.doc_id)
        await self.chat(prompt="What tools do you have access to? Make sure you use them going forward.")
        if not self.thread_id:
            self.thread_id = self.assistant_manager.thread.id
            self.update_agent_config_field( "thread_id", self.thread_id)
            logger.info(f"Thread id set to: {self.thread_id}")

        # Create modular subscriptions.
        subscriptions = []
        if self.doc_id:
            google_sub = GoogleDocSubscription(self, self.doc_id)
            subscriptions.append(google_sub)
        if self.channel_id:
            #list_channels_tool = self.tool_cache.get("slack_list_channels")
            #if not list_channels_tool:
            #    logger.error("Tool 'slack_list_channels' not found in cache.")
            #    return
            #result = call_tool_with_introspection(list_channels_tool, {"limit": 10})
            #output = json.loads(result["output"])
            #channels = [channel for channel in output.get('channels', [])
            #            if channel.get('name') == 'all-jake-test-ground']
            #if not channels:
            #    logger.error("No channel found with name 'all-jake-test-ground'.")
            #    return
            #channel_id = channels[0]['id']
            #slack_sub = SlackSubscription(self, channel_id)
            slack_sub = SlackSubscription(self, self.channel_id)
            subscriptions.append(slack_sub)

        tasks = [
            asyncio.create_task(self.cli_loop(self.doc_id)),
            *[asyncio.create_task(sub.poll()) for sub in subscriptions],
        ]
        await asyncio.gather(*tasks)

    async def create_doc(self):
        create_doc_tool = self.tool_cache.get("create-doc")
        result = call_tool_with_introspection(create_doc_tool, {"title": "New Doc"})
        logger.info(f"Assistant response: {result['output']}")
        doc_id = result["output"].split(":")[2].split("/")[5]
        return doc_id


# Helper to run an agent instance.
# If doc_id and thread_id are provided, re-instantiate that agent; otherwise, create a new document.
async def run_agent_instance(doc_id: str = None, assistant_id: str = None, done_assistant_id: str = None, thread_id: str = None, channel_id = None, mcps = None):
    if not mcps:
        mcps = []

    instructions = (
        "You are a long running document writing agent. "
        "You will be notified of doc changes and comments and your job is to update documents according to instructions. \n"
        "To edit the existing document you must wrap text content in backticks. \n\n"
        "The documents you work with will be divided into sections that are marked."
        "For the following examples lets work with the following document:\n"
        "###SECTION:1###\n"
        "section 1 content goes here\n"
        "###ENDSECTION###\n"
        "###SECTION:2###\n"
        "section 2 content goes here\n"
        "###ENDSECTION###\n\n"
        "When making changes to one of the sections, return your edits as a complete, updated text block(s) surrounded in backticks. "
        "If you include a section label (for example, 'section1') in the opening backticks, "
        "only that section will be updated. "
        "To update a section:\n"

        "```section1\n<updated text>\n```\n\n"

        "To update multiple sections write multiple blocks"
        "```section1\n<updated text 1>\n```\n\n"
        "```section3\n<updated text 2>\n```\n\n"

        "If you do not provide a section label (i.e. the opening backticks are empty)\n\n"
        "the entire document will be replaced with your output:\n\n"
        "To replace the entire doc:\n"
        "```\n<new full file content>\n```\n\n"

        "To add a new section at the end of the document pass a block with a *new* section id:\n"
        "```section3\nsection 3 text\n```\n\n"

        "By default, if a new section is created, it will be appended at the end.\n\n"

        "Feel free to move lines across section boundaries by replacing multiple boundaries if it helps make the section breaks more logical\n\n"
        "for example if you have\n"
        "```section1\na line about topic 1\nanother line about topic 1\nsomething about topic 2\n```\n\n"
        "```section2\nmore info on topic 2\n```\n\n"
        "modify it by sending\n"
        "```section1\na line about topic 1\nanother line about topic 1\n```\n\n"
        "```section2\nsomething about topic 2\nmore info on topic 2\n```\n\n"


        "You don't need to talk to the user about sections, they're purely there to help you make edits to the doc.\n"
        "When in doubt, ensure you are looking at the latest version of the doc.\n"
        "Don't use any markdown in the google doc.\n"
        "Always include URLs in your citations when referencing content from search tools.\n"
        "Only create new documents when specifically prompted to do so by the user. "
        "When you create documents make sure to share them with datastax.com. "
    )
    config_manager = ConfigManager(app_name="agentd", app_author="phact")
    agent = Agentd(
        instructions=instructions,
        config_manager=config_manager,
        mcps=mcps,
        doc_id=doc_id,
        channel_id=channel_id,
        assistant_id=assistant_id,
        done_assistant_id=done_assistant_id
    )
    if thread_id:
        agent.thread_id = thread_id
    await agent.run()


# Startup routine.
async def main():
    config_manager = ConfigManager(app_name="agentd", app_author="phact")
    config = config_manager.config
    tasks = []

    # Instantiate Slack-based agents
    for channel_id, agent_info in config.get("agents", {}).get("channels", {}).items():
        tasks.append(asyncio.create_task(
            run_agent_instance(
                doc_id=None,
                channel_id=channel_id,
                assistant_id=agent_info.get("assistant_id"),
                done_assistant_id=agent_info.get("done_assistant_id"),
                thread_id=agent_info.get("thread_id"),
                mcps=load_mcps_for("slack")
            )
        )
    )

    if len(tasks) == 0:
        tasks.append(asyncio.create_task(run_agent_instance(
            channel_id="C08H4QLK2RF",
            mcps=load_mcps_for("slack")
        )))

    # Instantiate document-based agents
    for doc_id, agent_info in config.get("agents", {}).get("documents", {}).items():
        tasks.append(
            asyncio.create_task(
                run_agent_instance(
                    doc_id=doc_id,
                    assistant_id=agent_info.get("assistant_id"),
                    done_assistant_id=agent_info.get("done_assistant_id"),
                    thread_id=agent_info.get("thread_id"),
                    mcps=load_mcps_for("documents")
                )
            )
        )


    await asyncio.gather(*tasks)

def load_mcps_for(subscription_type: str):
    brave_key = os.environ['BRAVE_API_KEY']
    mcps_mapping = {
        "documents": [
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
                tool_filter=['']
            ),
            #MCPRepresentationStdio(
            #    type="stdio",
            #    command="npx",
            #    arguments=["-y", "@modelcontextprotocol/server-brave-search"],
            #    env_vars=[f"BRAVE_API_KEY={brave_key}"],
            #),
            #MCPRepresentationStdio(
            #    type="stdio",
            #    command="uvx",
            #    arguments=["mcp-server-fetch"],
            #),
            #MCPRepresentationStdio(
            #    type="stdio",
            #    command="uv",
            #    arguments=["run", "--refresh", "--with", "mcp_simple_arxiv", "mcp-simple-arxiv"],
            #),
            MCPRepresentationStdio(
                type="stdio",
                command="npx",
                arguments=["-y", "@modelcontextprotocol/server-slack"],
                env_vars=[
                    f"SLACK_BOT_TOKEN={os.environ['SLACK_BOT_TOKEN']}",
                    f"SLACK_TEAM_ID={os.environ['SLACK_TEAM_ID']}",
                ],
                tool_filter=['']
            ),
            MCPRepresentationStdio(
                type="stdio",
                command="uv",
                arguments=["run","--refresh","--with","mcp_browser_use_tools", "mcp_server", "--headless"],
            )
        ],
        "slack": [
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
                tool_filter=['create-doc']
            ),
            MCPRepresentationStdio(
                type="stdio",
                command="npx",
                arguments=["-y", "@modelcontextprotocol/server-brave-search"],
                env_vars=[f"BRAVE_API_KEY={brave_key}"],
                tool_filter=['']
            ),
            MCPRepresentationStdio(
                type="stdio",
                command="uv",
                arguments=["run", "--refresh", "--with", "mcp_simple_arxiv", "mcp-simple-arxiv"],
                tool_filter=['']
            ),
            MCPRepresentationStdio(
                type="stdio",
                command="npx",
                arguments=["-y", "@modelcontextprotocol/server-slack"],
                env_vars=[
                    f"SLACK_BOT_TOKEN={os.environ['SLACK_BOT_TOKEN']}",
                    f"SLACK_TEAM_ID={os.environ['SLACK_TEAM_ID']}",
                ],
                tool_filter=['']
            )
        ]
    }
    return mcps_mapping.get(subscription_type, [])

if __name__ == "__main__":
    asyncio.run(main())
