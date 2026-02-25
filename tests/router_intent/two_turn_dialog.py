import json
import os
import random
import base64
import sys
from pydantic import BaseModel
from typing import Literal, Optional, List
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_app.agents.clarification.agent import clarification_node
from langgraph_app.orchestrator.state import GraphState, AgentInput
from langgraph_app.utils.gemini_client import GeminiClient
from collections import defaultdict

class TurnAction(BaseModel):
    needs_new_image: bool
    user_reply_text: str
    expected_intent: Literal["recognition", "recommendation", "clarification", "exit"]
    final_image_data_needed: bool # Does the final input for the second turn need an image?
    final_text_needed: bool      # Does the final input for the second turn need text?
    reasoning: str

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_random_food_image():
    food_dir = "tests/router_intent/images/food_img"
    if not os.path.exists(food_dir):
        return None
    files = [f for f in os.listdir(food_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        return None
    return os.path.join(food_dir, random.choice(files))

def generate_two_turn_cases(input_file, output_file, cases_per_category=2):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    with open(input_file, "r") as f:
        try:
            test_cases = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return

    client = GeminiClient()
    new_cases = []
    
    # Filter cases that need clarification and group by category
    clarification_cases_by_category = defaultdict(list)
    for case in test_cases:
        if case.get("expected_analysis", {}).get("intent") == "clarification":
            clarification_cases_by_category[case.get("category", "Unknown")].append(case)
    
    print(f"Found {sum(len(v) for v in clarification_cases_by_category.values())} clarification cases across categories. Processing...")

    for category, cases in clarification_cases_by_category.items():
        selected_cases = random.sample(cases, min(cases_per_category, len(cases)))
        
        for case in selected_cases:
            # 1. Run clarification node to get AI response
            case_input = case["input"]
            # Create an AgentInput instance
            agent_input = AgentInput(
                text=case_input.get("text", ""),
                image_data=case_input.get("image_data"),
                source=case_input.get("source", "user")
            )

            messages = []
            for msg in case["messages"]:
                if msg["type"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("type") in ["ai", "AI"]:
                    messages.append(AIMessage(content=msg["content"]))
            
            state: GraphState = {
                "input": agent_input,
                "messages": messages,
                "analysis": {
                    "intent": "clarification",
                    "safety_safe": True,
                    "safety_reason": None
                }
            }
            
            try:
                result_state = clarification_node(state)
                current_messages = result_state.get("messages", [])
                if not current_messages:
                    print(f'No messages in result state for ID {case["id"]}')
                    continue
                ai_msg = current_messages[-1]
                ai_response = ai_msg.content
            except Exception as e:
                print(f'Error running clarification node for ID {case["id"]}: {e}')
                import traceback
                traceback.print_exc()
                continue

            # 2. Use LLM to decide if we need a new image, user reply, and expected intent
            decision_prompt = f"""Based on the AI\"s clarification response, determine the next steps for the user\"s second turn.
            Consider the original user input and the AI\"s clarification.
            
            Original User Input Text: {agent_input['text']}
            AI Response: {ai_response}
            
            Instructions for determining TurnAction fields:
            1. 'needs_new_image': ALWAYS set to true if the AI's response indicates any issue with the current image (e.g., blurry, not food, missing, unclear) OR if the original input had no image but the AI implies one is needed for identification. If the AI is asking for more textual details about an existing, good image, then it can be false.
            2. 'user_reply_text': This should be a concise, direct user reply, mimicking real user behavior. Absolutely no apologetic or explanatory language (e.g., "I\"m sorry, I can\"t provide..."). IMPORTANT: If 'final_image_data_needed' is true AND 'final_text_needed' is false, this text MUST be an empty string, as the image itself is the primary input.
            3. 'expected_intent': The intent of the *second* user turn. This should be 'recognition' if a valid food image is expected, 'recommendation' if asking for food places/ideas, or 'clarification'/'exit' if the conversation continues to be unclear or ends.
            4. 'final_image_data_needed': Set to true if the *second* turn\"s input is expected to have an image (either new or reused good image). This should generally align with 'needs_new_image' if a new image is sought.
            5. 'final_text_needed': Set to true if the *second* turn\"s input is expected to have text.
            
            Respond with a JSON object conforming to the TurnAction schema."""

            try:
                action = client.generate_structured(decision_prompt, TurnAction)
            except Exception as e:
                print(f'Error getting LLM decision for ID {case["id"]}: {e}')
                action = TurnAction(needs_new_image=True, user_reply_text="", expected_intent="recognition", final_image_data_needed=True, final_text_needed=False, reasoning="Fallback")

            # 3. Prepare the second turn input based on LLM\"s decision
            second_turn_image_data = ""
            if action.final_image_data_needed:
                # If a new image is explicitly needed, or if the old one was empty but image data is still needed
                if action.needs_new_image or (not agent_input.get("image_data") and action.final_image_data_needed):
                    new_img_path = get_random_food_image()
                    if new_img_path:
                        second_turn_image_data = encode_image(new_img_path)
                else:
                    # Reuse the old image if still relevant and available
                    second_turn_image_data = agent_input.get("image_data") or ""
            
            second_turn_text = action.user_reply_text if action.final_text_needed else ""
            # Ensure text is empty if only image is needed (override if LLM returned text unintentionally)
            if action.final_image_data_needed and not action.final_text_needed:
                second_turn_text = "" 

            # 4. Construct the new case
            formatted_messages = []
            for m in current_messages:
                m_type = "human" if isinstance(m, HumanMessage) else "ai"
                formatted_messages.append({"type": m_type, "content": str(m.content)})
            
            # Add the second human reply
            formatted_messages.append({"type": "human", "content": action.user_reply_text})

            new_case = {
                "id": case["id"],
                "category": case["category"] + " - Two-Turn Clarification to " + action.expected_intent.capitalize(),
                "input": {
                    "text": second_turn_text,
                    "image_data": second_turn_image_data,
                    "source": "user"
                },
                "messages": formatted_messages,
                "expected_analysis": {
                    "intent": action.expected_intent,
                    "safety_safe": True,
                    "safety_reason": None
                }
            }
            new_cases.append(new_case)
            print(f'Processed case ID {case["id"]}: Expected Intent = {action.expected_intent}, New Image Needed = {action.needs_new_image}, Final Image Data Needed = {action.final_image_data_needed}, User Reply: "{action.user_reply_text}"')

    with open(output_file, "w") as f:
        json.dump(new_cases, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(new_cases)} two-turn cases to {output_file}")

if __name__ == "__main__":
    input_file = 'tests/router_intent/test_cases.json'
    output_file = 'tests/router_intent/test_cases_2.json'
    generate_two_turn_cases(input_file, output_file, cases_per_category=2)
