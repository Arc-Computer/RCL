from typing import Any, Dict, List, Union
import copy
import importlib.util
import shlex
import subprocess
import requests


class CustomWrapper:
    """Wrapper for any existing agent - API, CLI, Python function, etc."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integration_type = config["integration_type"]

        if self.integration_type == "python_function":
            self._setup_python_function()
        elif self.integration_type == "http_api":
            self.endpoint = config["endpoint"]
            self.headers = config.get("headers", {})
            self.request_template = config.get("request_template", {})
            self.prompt_field = config.get("prompt_field", "prompt")
            self.response_field = config.get("response_field", "response")
        elif self.integration_type == "cli_command":
            self.command = config["command"]

    def _setup_python_function(self):
        module_path = self.config["module_path"]
        function_name = self.config["function_name"]

        spec = importlib.util.spec_from_file_location("custom_agent", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.agent_function = getattr(module, function_name)

    def __call__(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        responses = []
        for prompt in prompts:
            response = self._call_agent(prompt)
            responses.append(response)

        return responses[0] if single else responses

    def _call_agent(self, prompt: str) -> str:
        if self.integration_type == "python_function":
            return self.agent_function(prompt)

        elif self.integration_type == "http_api":
            import copy
            payload = copy.deepcopy(self.request_template)
            payload[self.prompt_field] = prompt

            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=self.headers,
                    timeout=self.config.get("timeout", 300)
                )
                response.raise_for_status()
                result = response.json()
                return self._extract_field(result, self.response_field)
            except requests.exceptions.RequestException as e:
                return f"Error calling API: {str(e)}"
            except (KeyError, TypeError) as e:
                return f"Error parsing response: {str(e)}"

        elif self.integration_type == "cli_command":
            try:
                escaped_prompt = shlex.quote(prompt)
                cmd = self.command.replace("{prompt}", escaped_prompt)

                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.get("timeout", 300),
                    check=False
                )

                if result.returncode != 0:
                    return f"Command failed: {result.stderr}"
                return result.stdout.strip()
            except subprocess.TimeoutExpired:
                return "Command timed out"
            except Exception as e:
                return f"Command error: {str(e)}"

    def _extract_field(self, data: dict, field_path: str) -> str:
        try:
            for key in field_path.split("."):
                if isinstance(data, dict):
                    data = data.get(key)
                    if data is None:
                        return ""
                else:
                    return str(data)
            return str(data) if data is not None else ""
        except (AttributeError, TypeError):
            return ""