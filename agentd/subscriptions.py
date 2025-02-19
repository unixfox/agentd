import ast
import asyncio
import json
import time
import re
from datetime import datetime, timezone
from loguru import logger

class Subscription:
    def __init__(self, agent, identifier: str):
        """
        Base subscription class.

        :param agent: The Agentd instance.
        :param identifier: An identifier for this subscription
                           (for Google Docs this is the document id;
                           for Slack it might be a channel id)
        """
        self.agent = agent
        self.identifier = identifier

    async def poll(self):
        raise NotImplementedError("Subclasses must implement poll()")

class GoogleDocSubscription(Subscription):
    async def poll_comments(self):
        read_comments_tool = self.agent.tool_cache.get("read-comments")
        if not read_comments_tool:
            logger.error("Tool 'read-comments' not found in cache.")
            return

        last_time = 0
        while not self.agent.exit_event.is_set():
            try:
                result = self.agent.call_tool_with_introspection(
                    read_comments_tool, {"document_id": self.identifier}
                )
            except Exception as e:
                logger.error(f"Error calling read-comments tool: {e}")
                await asyncio.sleep(10)
                continue

            try:
                comments = ast.literal_eval(result["output"])
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
                    self.agent.interaction_origin = "comment"
                    self.agent.current_comment_id = comment.get("id")
                    await self.agent.reply_comment(self.identifier, comment["id"], "👀")
                    logger.info(f"New comment detected in document {self.identifier}.")
                    await self.agent.chat(
                        f"New comment detected in document {self.identifier}: {comment}."
                    )
                    last_time = time.time()
                    self.agent.new_comment_event.set()
            await asyncio.sleep(10)

    async def poll_doc_changes(self):
        read_doc_tool = self.agent.tool_cache.get("read-doc")
        if not read_doc_tool:
            logger.error("Tool 'read-doc' not found in cache.")
            return

        last_result = None
        last_prompt_time = 0
        while not self.agent.exit_event.is_set():
            try:
                result = self.agent.call_tool_with_introspection(
                    read_doc_tool, {"document_id": self.identifier}
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
                    await self.agent.chat(
                        f"Doc changes detected for doc {self.identifier}:\n{result}"
                    )
                    last_prompt_time = current_time
            await asyncio.sleep(10)

    async def poll(self):
        await asyncio.gather(self.poll_comments(), self.poll_doc_changes())

def markdown_to_slack(text):
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    def repl(match):
        return f'<{match.group(2)}|{match.group(1)}>'
    return re.sub(pattern, repl, text)


class SlackSubscription(Subscription):
    async def poll(self):
        slack_history_tool = self.agent.tool_cache.get("slack_get_channel_history")
        if not slack_history_tool:
            logger.error("Tool 'slack_get_channel_history' not found in cache.")
            return

        slack_post_message_tool = self.agent.tool_cache.get("slack_post_message")
        if not slack_post_message_tool:
            logger.error("Tool 'slack_post_message' not found in cache.")
            return

        slack_add_reaction_tool = self.agent.tool_cache.get("slack_add_reaction")
        if not slack_add_reaction_tool:
            logger.error("Tool 'slack_add_reaction' not found in cache.")
            return

        channel_id = self.identifier
        if not channel_id:
            logger.error("Slack channel id not provided in subscription identifier.")
            return

        last_timestamp = 0.0
        while not self.agent.exit_event.is_set():
            try:
                result = self.agent.call_tool_with_introspection(
                    slack_history_tool, {"channel_id": channel_id, "limit": 10}
                )
            except Exception as e:
                logger.error(f"Error calling slack_get_channel_history: {e}")
                await asyncio.sleep(10)
                continue

            try:
                messages = json.loads(result["output"])['messages']
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
                chat_result = await self.agent.chat(f"New Slack message: {text}")
                # Convert the chat response and include the bot marker.
                response_text = markdown_to_slack(chat_result['text'])
                text_for_post = f"[BOT COMMENT]:\n{response_text}"
                self.agent.call_tool_with_introspection(
                    slack_post_message_tool,
                    {"channel_id": channel_id, "text": text_for_post}
                )
                # Mark the message as processed by adding an "eyes" reaction.
                reaction_return = self.agent.call_tool_with_introspection(
                    slack_add_reaction_tool,
                    {"channel_id": channel_id, "timestamp": msg.get("ts"), "reaction": "eyes"}
                )
                last_timestamp = ts
            await asyncio.sleep(10)
