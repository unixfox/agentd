import ast
import asyncio
import json
import os
import time
import re
from datetime import datetime, timezone
from typing import List

from loguru import logger

from agentd.section_chunker import unwrap_sections
from agentd.tool_util import call_tool_with_introspection

from agents.mcp import MCPServer, MCPServerStdio

from mcp.types import Tool as MCPTool
from agents import Agent, Runner


class Subscription:
    def __init__(self, agent: Agent, identifier: str, current_response_id=None):
        """
        Base subscription class.

        :param agent: The Agentd instance.
        :param identifier: An identifier for this subscription
                           (for Google Docs this is the document id;
                           for Slack it might be a channel id)
        """
        self.agent = agent
        self.identifier = identifier
        self.current_response_id = current_response_id

    async def poll(self):
        raise NotImplementedError("Subclasses must implement poll()")


    async def connect_mcp(self):
        raise NotImplementedError("Subclasses must implement poll()")


class GoogleDocSubscription(Subscription):

    async def connect_mcp(self):
        await self.mcp_server.connect()

    async def poll_comments(self):
        read_comments_tool = "read-comments"

        last_time = 0
        while True:
            try:
                result = await self.mcp_server.call_tool(
                    read_comments_tool,
                    {
                        "document_id": self.identifier
                    }
                )
            except Exception as e:
                logger.error(f"Error calling read-comments tool: {e}")
                await asyncio.sleep(10)
                continue

            try:
                #comments = ast.literal_eval(result.content)
                comments = result.content
            except Exception as e:
                logger.error(f"Error parsing comments: {e}")
                await asyncio.sleep(10)
                continue

            for comment in comments:
                try:
                    modifiedTime = datetime.strptime(
                        comment["modifiedTime"][:-1], "%Y-%m-%dT%H:%M:%S.%f"
                    ).replace(tzinfo=timezone.utc).timestamp()
                except Exception:
                    continue
                if comment.get('resolved'):
                    continue
                if modifiedTime > last_time:
                    if ("replies" in comment and comment["replies"] and
                            comment["replies"][-1]["content"].startswith("[BOT COMMENT]")):
                        continue

                    logger.info(f"New comment detected in document {self.identifier}.")

                    #eyes
                    await self.mcp_server.call_tool(
                        "reply-comment",
                        {
                            "document_id": self.identifier,
                            "comment_id": comment["id"],
                            "reply": f"[BOT COMMENT]:\n👀"
                        }
                    )

                    msg =  (
                        f"New comment detected in document {self.identifier}: {comment}.\n"
                        f"The current as displayed to the user is: \n{unwrap_sections(self.agent.doc_state)}\n"
                        f"The current with sections for editing is: \n{self.agent.doc_state}\n"
                        "First make a plan and perform any relevant function calls that may apply to your task, "
                        "then proceed to make the document edits."
                    )
                    result = await Runner.run(self.agent, msg)
                    print(result)

                    # TODO - update the doc and comment?

                    last_time = time.time()
            await asyncio.sleep(10)

    async def poll_doc_changes(self):
        read_doc_tool = "read-doc"

        last_result = None
        last_prompt_time = 0
        while True:
            try:
                result = await self.mcp_server.call_tool(
                    read_doc_tool,
                    {"document_id": self.identifier}
                )
            except Exception as e:
                logger.error(f"Error calling read-doc tool: {e}")
                await asyncio.sleep(10)
                continue

            if last_result is None:
                last_result = result
            elif last_result != result:
                last_result = result
                current_time = time.time()
                if current_time - last_prompt_time > 30:
                    logger.info(f"Detected change in document {self.identifier}.")
                    msg = f"Doc changes detected for doc {self.identifier}:\n{result}"
                    result = Runner.run(self.agent, msg)
                    print(result)
                    last_prompt_time = current_time
            await asyncio.sleep(10)

    mcp_server = MCPServerStdio(
            name="GoogleDocs",
            params={
                "command": "uv",
                "args": [
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
            },
            cache_tools_list=True
        )

    async def poll(self):
        await asyncio.gather(self.poll_comments(), self.poll_doc_changes())


def markdown_to_slack(text):
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    def repl(match):
        return f'<{match.group(2)}|{match.group(1)}>'

    return re.sub(pattern, repl, text)


class SlackSubscription(Subscription):
    async def connect_mcp(self):
        await self.mcp_server.connect()

    async def poll(self):
        slack_history_tool = "slack_get_channel_history"
        slack_post_message_tool = "slack_post_message"
        slack_add_reaction_tool = "slack_add_reaction"

        channel_id = self.identifier
        if not channel_id:
            logger.error("Slack channel id not provided in subscription identifier.")
            return

        last_timestamp = 0.0
        while True:
            try:
                result = await self.mcp_server.call_tool(
                    slack_history_tool,
                    {
                        "channel_id": channel_id,
                        "limit": 10
                    }
                )
            except Exception as e:
                logger.error(f"Error calling slack_get_channel_history: {e}")
                await asyncio.sleep(10)
                continue

            try:
                messages = json.loads(result.content[0].text)['messages']
            except Exception as e:
                logger.error(f"Error parsing Slack history: {e}")
                await asyncio.sleep(10)
                continue

            if messages[0]['text'].startswith("[BOT COMMENT]:"):
                await asyncio.sleep(10)
                continue
            for msg in messages:
                try:
                    ts = float(msg.get("ts", "0"))
                except Exception:
                    continue

                text = msg.get("text", "").strip()
                # Skip processing if the message already contains the bot marker.
                if text.startswith("[BOT COMMENT]:"):
                    continue

                # Ignore messages older than the most recent bot response.
                if ts <= last_timestamp:
                    continue

                logger.info(f"New Slack message detected: {text}")
                # Mark the message as processing by adding an "eyes" reaction.
                await self.mcp_server.call_tool(
                    slack_add_reaction_tool,
                    {
                        "channel_id": channel_id,
                        "timestamp": msg.get("ts"),
                        "reaction": "eyes"
                    }
                )

                chat_result = await Runner.run(self.agent, f"New Slack message: {text}")
                # Convert the chat response and include the bot marker.
                response_text = markdown_to_slack(chat_result['text'])
                text_for_post = f"[BOT COMMENT]:\n{response_text}"

                await self.mcp_server.call_tool(
                    slack_post_message_tool,
                    {
                        "channel_id": channel_id,
                        "text": text_for_post
                    }
                )

                last_timestamp = ts
            await asyncio.sleep(10)


    mcp_server = MCPServerStdio(
            name="Slack",
            params={
                "command": "npx",
                "args": [
                    "-y", "@modelcontextprotocol/server-slack"
                ],
                "env": {
                    "SLACK_BOT_TOKEN": f"{os.environ['SLACK_BOT_TOKEN']}",
                    "SLACK_TEAM_ID": f"{os.environ['SLACK_TEAM_ID']}"
                }
            },
            cache_tools_list=True
        )

