# test_assistant_integration.py
import os
import pytest

from agentd.ui.assistant_util import create_manager, list_threads, list_messages

# Define a list of required environment variables for the integration tests.
REQUIRED_ENV_VARS = ["ASTRA_DB_APPLICATION_TOKEN"]

@pytest.fixture(scope="module")
def manager():
    # Check that all required env vars are set.
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        pytest.skip(f"Skipping integration tests because these env vars are missing: {missing}")
    # Create an AssistantManager with real credentials and instructions.
    return create_manager("Integration Test Assistant", model="gpt-4o")

def test_list_threads_integration(manager):
    threads = list_threads(manager)
    # Expect a non-empty list of threads.
    assert isinstance(threads, list)
    assert len(threads) > 0
    for thread in threads:
        # Each thread should have an 'id' attribute.
        assert hasattr(thread, "id")
        assert isinstance(thread.id, str)
        for message in thread.messages:
            print(message)

def test_list_messages_integration(manager):
    threads = list_threads(manager)
    # Pick the first thread from the list.
    thread = threads[0]
    messages = list_messages(manager, thread.id)
    # messages should be a list (could be empty if no messages yet).
    assert isinstance(messages, list)
