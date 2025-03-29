import asyncio
import os
from typing import List, Dict

from agents import Agent, Runner
from openai import AsyncOpenAI
from agents.items import TResponseInputItem


from agentd.config import ConfigManager
from agentd.models.message import create_message, Message, ContentItem

# Initialize the OpenAI client (using Responses API)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


async def run_agent(prompt: str, response_id: str = None, config_manager: ConfigManager = None):
    # Load the conversation history from the response id.
    conversation_history = []
    if response_id:
        conversation_history = await load_conversation_history(response_id)

    # Create an agent with your instructions.
    agent = Agent(
        name="Contextual Agent",
        instructions="You are an assistant that uses prior conversation context to provide informed responses."
    )

    # Pass the conversation history as context when running the agent.
    # Note: the 'context' parameter is a user-defined payload that your agent can inspect.
    messages = conversation_history + [
        Message(role="user", content=[ContentItem(text=prompt, type="input_text")], status=None).model_dump(),
    ]
    result = await Runner.run(starting_agent=agent, input=messages, context={"conversation_history": conversation_history})
    return result

async def main():

    config_manager = ConfigManager(app_name="agentd", app_author="phact")
    config = config_manager.config
    tasks = []

    prompt = "hi"

    CHANNELS = "channels"
    DOCUMENTS = "documents"
    AGENTS = "agents"
    # Instantiate Slack-based agents
    for channel_id, agent_info in config.get(AGENTS, {}).get(CHANNELS, {}).items():
        response_id = agent_info.get("response_id", None)
        result = await run_agent(response_id = response_id, prompt = prompt)
        response_id = result.raw_responses[-1].referenceable_id
        config_manager.update_agent_config_field("response_id", response_id, "channels", channel_id)

    if len(tasks) == 0:
        channel_id = "C08LJBEM46L"
        agent_type = CHANNELS
        result = await run_agent(prompt = prompt)
        response_id = result.raw_responses[-1].referenceable_id
        config_manager.update_agent_config_field("response_id", response_id, agent_type, channel_id)

    # Instantiate document-based agents
    for doc_id, agent_info in config.get(AGENTS, {}).get(DOCUMENTS, {}).items():
        response_id = agent_info.get("response_id", None)
        result = await run_agent(response_id = response_id, prompt = prompt)
        response_id = result.raw_responses[-1].referenceable_id
        config_manager.update_agent_config_field("response_id", response_id, DOCUMENTS, doc_id)


    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())