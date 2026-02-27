import os
from langgraph_app.config import Config

def test_sampling_params():
    config = Config()
    
    # 1. Test Default
    params = config.get_sampling_params("openai", "router")
    print(f"Default (OpenAI, Router): {params}")
    assert params["temperature"] == 0.15
    assert params["top_p"] == 0.85
    
    # 2. Test Module Override (already in config)
    params = config.get_sampling_params("openai", "clarification")
    print(f"Module (OpenAI, Clarification): {params}")
    assert params["temperature"] == 0.35
    assert params["presence_penalty"] == 0.1
    
    # 3. Test Environment Variable Override
    os.environ["LLM_SAMPLING_OPENAI_ROUTER_TEMPERATURE"] = "0.99"
    os.environ["LLM_SAMPLING_OPENAI_ROUTER_TOP_P"] = "0.11"
    
    params = config.get_sampling_params("openai", "router")
    print(f"Env Override (OpenAI, Router): {params}")
    assert params["temperature"] == 0.99
    assert params["top_p"] == 0.11
    
    # Clean up
    del os.environ["LLM_SAMPLING_OPENAI_ROUTER_TEMPERATURE"]
    del os.environ["LLM_SAMPLING_OPENAI_ROUTER_TOP_P"]
    
    print("All sampling param tests passed!")

if __name__ == "__main__":
    test_sampling_params()