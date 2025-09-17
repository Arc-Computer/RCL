import json
from typing import Any, Dict, List, Union

try:
    from agents import Agent, Runner
except ImportError:
    raise ImportError(
        "OpenAI Agents SDK not found. Install it with: "
        "pip install openai-agents"
    )


class OpenAISDKWrapper:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        return Agent(
            name=self.config.get("name", "Agent"),
            instructions=self.config.get("instructions", ""),
            tools=self._load_tools(),
            handoffs=self._load_handoffs(),
            model=self.config.get("model"),
        )

    def _load_tools(self):
        tools = []
        for tool_config in self.config.get("tools", []):
            if tool_config["type"] == "function":
                from agents import function_tool
                import importlib.util

                if "module_path" in tool_config:
                    spec = importlib.util.spec_from_file_location(
                        "custom_tool",
                        tool_config["module_path"]
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    func = getattr(module, tool_config["function_name"])
                    tools.append(function_tool(func))
        return tools

    def _load_handoffs(self):
        handoffs = []
        for handoff_config in self.config.get("handoffs", []):
            handoff_agent = Agent(
                name=handoff_config["name"],
                instructions=handoff_config["instructions"],
            )
            handoffs.append(handoff_agent)
        return handoffs

    def __call__(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        responses = []
        for prompt in prompts:
            result = Runner.run_sync(self.agent, prompt)
            output = result.final_output

            if isinstance(output, dict):
                output = json.dumps(output)
            elif not isinstance(output, str):
                output = str(output)

            responses.append(output)

        return responses[0] if single else responses