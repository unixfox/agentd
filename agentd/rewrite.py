import asyncio
import contextlib
import functools
import os
from typing import List, Dict, Type

from agents import Agent, Runner
from openai import AsyncOpenAI
from agents.items import TResponseInputItem

from agentd.config import ConfigManager
from agentd.models.message import create_message, Message, ContentItem
from agentd.subscriptions import Subscription, SlackSubscription, GoogleDocSubscription

# Initialize the OpenAI client (using Responses API)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def with_async_exit_stack(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        async with contextlib.AsyncExitStack() as stack:
            # Pass the exit_stack to the wrapped function
            return await func(*args, exit_stack=stack, **kwargs)
    return wrapper

async def load_conversation_history(response_id: str) -> List[Dict[str, any]]:
    """
    Loads the conversation history associated with a given response_id.
    """
    item_list = await client.responses.input_items.list(response_id)
    response = await client.responses.retrieve(response_id)
    latest_message = response.output if hasattr(response, "output") else response.get("output", "")
    full_message_history: List[TResponseInputItem] = item_list.data + latest_message
    serializable_messages = [create_message(msg).model_dump() for msg in full_message_history]
    return serializable_messages

async def setup_agent(sub_type_cls: Type[Subscription], subscribe_to_id, name, instructions, response_id=None):
    # Load the conversation history if available.

    # Create an agent with your instructions.
    agent = Agent(
        name=name,
        instructions=instructions,
    )

    subscription = sub_type_cls(
        agent=agent,
        identifier=subscribe_to_id,
        current_response_id=response_id
    )
    await subscription.connect_mcp()
    return subscription


async def run_agent(prompt: str, response_id: str = None, agent: Agent = None, config_manager: ConfigManager = None):
    conversation_history = []
    if response_id:
        conversation_history = await load_conversation_history(response_id)

    messages = conversation_history + [
        Message(role="user", content=[ContentItem(text=prompt, type="input_text")], status=None).model_dump(),
    ]
    result = await Runner.run(starting_agent=agent, input=messages, context={"conversation_history": conversation_history})
    return result

class AgentD:
    """
    AgentD holds the configuration and sets up subscriptions for each agent.
    It parses the config and kicks off the agent subscriptions.
    """
    def __init__(self, prompt: str = "hi"):
        self.prompt = prompt
        # Create a config manager instance.
        self.config_manager = ConfigManager(app_name="agentd", app_author="phact")
        self.config = self.config_manager.config

        self.subscriptions : List[Subscription] = []
        self.subscription_tasks : List[asyncio.Task] = []

    async def start_subscriptions(self):
        AGENTS = "agents"
        CHANNELS = "channels"
        DOCUMENTS = "documents"
        INSTRUCTIONS = "You're a heplful assistant"

        # For channel-based (e.g., Slack) agents:
        channels_config = self.config.get(AGENTS, {}).get(CHANNELS, {})
        if channels_config:
            for channel_id, agent_info in channels_config.items():
                response_id = agent_info.get("response_id", None)
                # Kick off the subscription for this channel.
                subscription = await setup_agent(
                    sub_type_cls=SlackSubscription,
                    subscribe_to_id=channel_id, 
                    name=f"slack-agent{channel_id}", 
                    instructions=INSTRUCTIONS,
                    response_id=response_id
                )
                self.subscriptions.append(subscription)
                task = subscription.poll()
                self.subscription_tasks.append(task)
        else:
            # If there are no channel subscriptions configured, set up a default one.
            channel_id = "C08LJBEM46L"
            subscription = await setup_agent(
                sub_type_cls=SlackSubscription,
                subscribe_to_id=channel_id,
                name=f"slack-agent{channel_id}",
                instructions=INSTRUCTIONS,
            )
            self.subscriptions.append(subscription)
            task = subscription.poll()
            self.subscription_tasks.append(task)

            #result = await setup_agent(prompt=self.prompt)
            #response_id = result.raw_responses[-1].referenceable_id
            #self.config_manager.update_agent_config_field("response_id", response_id, CHANNELS, channel_id)

        # For document-based agents:
        documents_config = self.config.get(AGENTS, {}).get(DOCUMENTS, {})
        for doc_id, agent_info in documents_config.items():
            response_id = agent_info.get("response_id", None)
            subscription = await setup_agent(
                sub_type_cls=GoogleDocSubscription,
                subscribe_to_id=doc_id,
                name=f"{DOCUMENTS}-agent{doc_id}",
                instructions=INSTRUCTIONS,
                response_id=response_id
            )
            self.subscriptions.append(subscription)
            task = subscription.poll()
            self.subscription_tasks.append(task)

        # Await channel subscriptions concurrently.
        if self.subscription_tasks:
            await asyncio.gather(
                *self.subscription_tasks
            )

    async def run(self):
        # Start all subscriptions.
        await self.start_subscriptions()

# Main entry point
async def main():
    agentd_instance = AgentD(prompt="hi")
    await agentd_instance.run()

if __name__ == "__main__":
    asyncio.run(main())
