from typing import Any, Callable, Dict, Union


def load_wrapper(wrapper_type: str, config: Dict[str, Any]) -> Callable:
    if wrapper_type == "openai_sdk":
        from .openai_sdk_wrapper import OpenAISDKWrapper
        return OpenAISDKWrapper(config)
    elif wrapper_type == "openai_assistant":
        from .openai_assistant_wrapper import OpenAIAssistantWrapper
        return OpenAIAssistantWrapper(config)
    elif wrapper_type == "custom":
        from .custom_wrapper import CustomWrapper
        return CustomWrapper(config)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")