import operator
import re
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END

# 1. Define State
class AgentState(TypedDict):
    input: str
    agent_scratchpad: str
    final_answer: str
    next_action: str
    next_action_args: str
    steps: Annotated[List[str], operator.add]

def create_react_graph(llm, tools):
    # Prepare tool details for the prompt
    tool_names = ", ".join([t.name for t in tools])
    tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in tools])
    
    # Standard ReAct Prompt
    prompt_template = f"""Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format strictly:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}"""

    # 2. ReAct Node (Reasoning + Action)
    async def react_node(state: AgentState):
        prompt = prompt_template.format(
            input=state["input"],
            agent_scratchpad=state["agent_scratchpad"]
        )
        
        # Async invoke the local model
        response = await llm.ainvoke(prompt)
        content = response.content
        
        new_scratchpad = state.get("agent_scratchpad", "") + content
        
        # Parse for Final Answer
        final_match = re.search(r"Final Answer:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
        if final_match:
            return {
                "agent_scratchpad": new_scratchpad,
                "final_answer": final_match.group(1).strip(),
                "steps": [content]
            }
            
        # Parse for Action and Action Input
        action_match = re.search(r"Action:\s*(.*?)\n", content, re.IGNORECASE)
        action_input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", content, re.IGNORECASE)
        
        if action_match and action_input_match:
            return {
                "agent_scratchpad": new_scratchpad,
                "next_action": action_match.group(1).strip(),
                "next_action_args": action_input_match.group(1).strip(),
                "steps": [content]
            }
            
        # Fallback if model breaks format
        return {
            "agent_scratchpad": new_scratchpad + "\nObservation: Invalid format. Please provide 'Action:' and 'Action Input:' or 'Final Answer:'.\nThought: ",
            "steps": ["Observation: Format Error"]
        }

    # 3. Tool Execution Node
    async def tool_node(state: AgentState):
        action = state.get("next_action")
        action_input = state.get("next_action_args")
        
        tools_map = {t.name: t for t in tools}
        
        if action in tools_map:
            try:
                # Clean up quotes from the arguments
                clean_args = action_input.strip("\"'")
                # Using ainvoke because MCP tools are asynchronous
                observation = str(await tools_map[action].ainvoke(clean_args))
            except Exception as e:
                observation = f"Error executing {action}: {e}"
        else:
            observation = f"Tool '{action}' not found."
            
        new_scratchpad = state.get("agent_scratchpad", "") + f"\nObservation: {observation}\nThought: "
        
        return {
            "agent_scratchpad": new_scratchpad,
            "steps": [f"Observation: {observation}"]
        }

    # 5. Conditional Routing logic
    def route_node(state: AgentState):
        if state.get("final_answer"):
            return END
        return "tool_node"

    # 4. Graph Flow compilation
    workflow = StateGraph(AgentState)
    workflow.add_node("react_node", react_node)
    workflow.add_node("tool_node", tool_node)
    
    workflow.set_entry_point("react_node")
    workflow.add_conditional_edges("react_node", route_node)
    workflow.add_edge("tool_node", "react_node")
    
    return workflow.compile()### Graph
