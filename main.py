import os
import sys
import asyncio
import nest_asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from graph import create_react_graph

# Apply nest_asyncio to prevent Windows event loop crashes
nest_asyncio.apply()

async def run_agent():
    print("--- Connecting to MCP Servers ---")
    
    # Path configuration for the tools folder
    tools_dir = os.path.abspath("Tools")
    math_path = os.path.join(tools_dir, "math_server.py")
    search_path = os.path.join(tools_dir, "search_server.py")

    # Configure the MCP Client mapping to your project structure
    mcp = MultiServerMCPClient({
        "math": {
            "command": sys.executable,
            "args": [math_path],
            "transport": "stdio",
        },
        "search": {
            "command": sys.executable,
            "args": [search_path],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    })

    tools = []
    
    try:
        print(" -> Connecting to Math server...")
        tools.extend(await mcp.get_tools(server_name="math"))
        
        print(" -> Connecting to Search server...")
        tools.extend(await mcp.get_tools(server_name="search"))

        print(" -> Connecting to Weather server...")
        tools.extend(await mcp.get_tools(server_name="weather"))
        
        print(f"✅ All MCP tools loaded successfully: {[t.name for t in tools]}\n")
    except Exception as e:
        print(f"❌ FATAL ERROR loading MCP tools: {e}")
        print("CRITICAL: Make sure all dependencies are installed and the weather server is running.")
        return # Stop execution if tools fail to load

    # Setup Local LLM
    llm = ChatOllama(model="qwen3.5:2b", temperature=0)
    
    # Compile Graph
    app = create_react_graph(llm, tools)
    
    # Test Case defined in requirements
    query = (
        "What is the weather in Lahore and who is the current Prime Minister of Pakistan? "
        "Now get the age of PM and tell us will this weather suits PM health."
    )
    
    print(f"--- STARTING QUERY ---\n{query}\n")
    
    initial_state = {
        "input": query,
        "agent_scratchpad": "",
        "final_answer": "",
        "next_action": "",
        "next_action_args": "",
        "steps": []
    }

    # Stream the graph execution
    final_state_snapshot = {} # Variable to hold the last state
    # Stream the graph execution
    async for output in app.astream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n[--- Node: {node_name} ---]")
            if state_update.get("steps"):
                # Print the most recent step/reasoning string
                print(state_update["steps"][-1])
        # Capture the state from the last node's output
        final_state_snapshot = list(output.values())[0] 

    print("\n================ FINAL ANSWER ================\n")
    # Print the final answer from the captured last state
    print(final_state_snapshot.get("final_answer", "No final answer reached."))

if __name__ == "__main__":
    asyncio.run(run_agent())
