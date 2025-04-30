# agentd

View sample configs in the [`config/`](https://github.com/phact/agentd/tree/main/config) directory

### to run:

    uvx agentd config.yaml

Built using [mcp-subscribe](https://github.com/phact/mcp-subscribe) a way for agents to subscribe to any MCP's tools

### MCP with chat completions

Building this I realized building an MCP Client with which to use your servers is too complex. Enter `patch_openai_with_mcp`.

Just wrap the openai client and pass it `mcp_servers=[…]`. You can use the `MCPServerStdio` or `MCPServerSSE` classes from openai-agents. I'm not sure why openai hadn't done this yet to be honest so I decided to scratch my own itch.

Example:

```python
from agents.mcp.server import MCPServerStdio
from agentd.patch import patch_openai_with_mcp
from openai import OpenAI

fs_server = MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/"],
    },
    cache_tools_list=True
)

client = patch_openai_with_mcp(OpenAI())  # Get the patched client back

response = client.chat.completions.create(
    #model="gpt-4o",
    #model="claude-3-5-sonnet-20240620",
    model="gemini/gemini-2.0-flash",
    messages=[
        {"role": "user", "content": "List the files in /tmp/ using the tool"}
    ],
    mcp_servers=[fs_server],
    mcp_strict=True
)

print(response.choices[0].message.content)
```
