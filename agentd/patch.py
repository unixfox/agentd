# openai_mcp_patch.py

import asyncio
import json
from functools import wraps
from openai.resources.chat.completions import Completions, AsyncCompletions
from agents.mcp.util import MCPUtil

# Global server cache
_SERVER_CACHE = {}

async def _ensure_connected(server):
    """Return the same connected server every time (stdio or sse)."""
    if server.name not in _SERVER_CACHE:
        await server.connect()
        _SERVER_CACHE[server.name] = server
    return _SERVER_CACHE[server.name]

def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)

def patch_openai_with_mcp(client):
    """
    Monkey-patch both sync and async chat.completions.create to:
      - accept mcp_servers=[…] & mcp_strict flag
      - fetch tools via MCPUtil.get_all_function_tools
      - resolve LLM provider via LiteLLM
      - inject headers/base_url accordingly

    Args:
        client: OpenAI or AsyncOpenAI client instance

    Returns:
        The patched client instance
    """
    is_async = client.__class__.__name__ == 'AsyncOpenAI'

    # Store original methods before patching
    orig_sync = Completions.create
    orig_async = AsyncCompletions.create

    async def _prepare(servers, strict):
        """Ensure servers are connected and get their tools."""
        connected = [await _ensure_connected(s) for s in servers]
        tools = await MCPUtil.get_all_function_tools(connected, strict)
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.params_json_schema
            }
        } for tool in tools]

    def _clean_kwargs(kwargs):
        """Remove our custom kwargs."""
        kwargs = kwargs.copy()
        kwargs.pop('mcp_servers', None)
        kwargs.pop('mcp_strict', None)
        return kwargs


    # constants
    MAX_TOOL_LOOPS = 5          # safety valve

    async def _handle_completion(
            self, args, model, messages,
            mcp_servers, mcp_strict, tools, kwargs, async_mode
    ):
        if mcp_servers and tools:
            raise ValueError("Cannot specify both mcp_servers and tools")

        if mcp_servers:
            tools = await _prepare(mcp_servers, mcp_strict)

        server_lookup = {t.name: s
                         for s in mcp_servers
                         for t in await s.list_tools()}

        clean_kwargs = _clean_kwargs(kwargs)

        if tools and "tool_choice" not in clean_kwargs:
            clean_kwargs["tool_choice"] = "auto"

        # ----- loop until no tool_calls -----
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > MAX_TOOL_LOOPS:
                raise RuntimeError("Tool-call loop exceeded MAX_TOOL_LOOPS")

            # ----- make the chat completion call -----
            resp = (
                await orig_async(self, *args, model=model, messages=messages,
                                 tools=tools, **clean_kwargs)
                if async_mode else
                orig_sync(self, *args, model=model, messages=messages,
                          tools=tools, **clean_kwargs)
            )

            tcalls = resp.choices[0].message.tool_calls

            if not tcalls:
                # no more tool calls → done
                return resp

            # consume every tool_call returned in *this* response
            for call in tcalls:
                tname = call.function.name
                targ_json = (call.function.arguments)
                if not isinstance(targ_json, dict):
                    targ_json = json.loads(targ_json)

                server = server_lookup.get(tname)
                if not server:
                    raise KeyError(f"Tool '{tname}' not found in server lookup")

                # run the tool via MCP
                result_obj = await server.call_tool(tname, targ_json)

                # append the assistant call + tool response
                messages = messages + [
                    {"role": "assistant", "tool_calls": [call]},
                    {"role": "tool",
                     "name": tname,
                     "content": result_obj.dict().get("content"),
                     "tool_call_id": call["id"] if isinstance(call, dict) else call.id},
                ]

            #clean_kwargs.pop("tools", None)
            #clean_kwargs.pop("tool_choice", None)
            #tools = None   # subsequent turns shouldn't resend schemas

    @wraps(Completions.create)
    def patched_sync(self, *args, model=None, messages=None,
                     mcp_servers=None, mcp_strict=False, tools=None, **kwargs):
        return _run_async(_handle_completion(self, args, model, messages,
                                          mcp_servers, mcp_strict, tools, kwargs, False))

    @wraps(AsyncCompletions.create)
    async def patched_async(self, *args, model=None, messages=None,
                           mcp_servers=None, mcp_strict=False, tools=None, **kwargs):
        return await _handle_completion(self, args, model, messages,
                                     mcp_servers, mcp_strict, tools, kwargs, True)

    # Patch only the appropriate method based on client type
    if is_async:
        AsyncCompletions.create = patched_async
    else:
        Completions.create = patched_sync

    return client
