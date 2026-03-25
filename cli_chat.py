import os
import re
import sys
import time
import base64
import queue
import threading
import argparse
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Log level configuration
LOG_VIEW_ALL = "all"
LOG_VIEW_NODE = "node"
LOG_VIEW_NONE = "none"

log_view_mode = LOG_VIEW_ALL  # default

# Import the graph
from langgraph_app.orchestrator.graph import graph

# Input queue for the user stack
user_queue = queue.Queue()
# To keep track of history since the graph does not have memory compiled in by default
chat_history = []


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".webp":
        return "image/webp"
    return "image/jpeg"  # default


def parse_user_input(text: str) -> list:
    """
    Parse the user input to find image paths formatted as {path}.
    Returns a list of content dicts compatible with HumanMessage.
    """
    pattern = r"\{([^}]+)\}"
    matches = list(re.finditer(pattern, text))

    content_list = []
    last_end = 0

    for match in matches:
        start, end = match.span()
        # Add text before the image
        text_part = text[last_end:start].strip()
        if text_part:
            content_list.append({"type": "text", "text": text_part})

        # Process the image path
        img_path = match.group(1).strip()
        if os.path.isfile(img_path):
            try:
                base64_img = encode_image_to_base64(img_path)
                mime_type = get_image_mime_type(img_path)
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
                    }
                )
                print(f"[System] Successfully loaded image: {img_path}")
            except Exception as e:
                print(f"[System] Error loading image {img_path}: {e}")
                content_list.append({"type": "text", "text": match.group(0)})
        else:
            print(
                f"[System] Warning: Image path not found: {img_path}. Treating as raw text."
            )
            content_list.append({"type": "text", "text": match.group(0)})

        last_end = end

    # Add remaining text
    remaining_text = text[last_end:].strip()
    if remaining_text:
        content_list.append({"type": "text", "text": remaining_text})

    # If no complex content, just return the string (or a single text dict)
    if not content_list:
        return [{"type": "text", "text": text}]

    return content_list


def input_thread_func():
    """Thread to continuously read user input and push to the queue."""
    print("==========================================================")
    print("CLI Chat Debug Interface Started")
    print("Type your message and press Enter.")
    print("To include an image, use the format: {/path/to/image.jpg}")
    print("Multiple inputs will be queued and processed sequentially.")
    print("Type 'exit' or 'quit' to stop.")
    print("==========================================================")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                user_queue.put(None)  # Sentinel value to exit
                break
            if user_input.strip():
                user_queue.put(user_input)
        except EOFError:
            user_queue.put(None)
            break
        except KeyboardInterrupt:
            user_queue.put(None)
            break


def process_queue():
    """Main loop to process messages from the queue."""
    global chat_history
    session_id = f"cli_session_{int(time.time())}"

    while True:
        # Wait for the next input
        user_input = user_queue.get()

        if user_input is None:
            print("[System] Exiting...")
            break

        print("\n[System] Processing your input...")

        # Parse the input for text and images
        content = parse_user_input(user_input)

        # Create the HumanMessage
        # If content has only 1 text element, we can simplify it
        if len(content) == 1 and content[0]["type"] == "text":
            new_msg = HumanMessage(content=content[0]["text"])
        else:
            new_msg = HumanMessage(content=content)

        # Prepare state
        # We append our new message to the chat history to provide context
        current_messages = chat_history + [new_msg]

        initial_state: dict = {
            "messages": current_messages,
            "session_id": session_id,
            "patient_id": "cli_test_user",
        }

        try:
            # Invoke the graph
            # Note: Depending on your graph implementation, it might take a while.
            result = graph.invoke(initial_state)  # type: ignore

            # The result should contain the updated messages
            if "messages" in result and result["messages"]:
                # Update our chat history to match the latest state
                chat_history = result["messages"]

                # The last message is usually the AI response
                last_msg = chat_history[-1]
                if isinstance(last_msg, AIMessage):
                    print(f"\nAI: {last_msg.content}\n")
                else:
                    # In case the last message is not an AIMessage for some reason
                    print(f"\nLast State Message: {last_msg.content}\n")
            else:
                print("\n[System] No messages returned in state.\n")

        except Exception as e:
            print(f"\n[Error] Failed to process through LangGraph: {e}\n")
            # If it fails, we shouldn't add the broken interaction to history

        finally:
            user_queue.task_done()
            # Let the user know we're ready for the next queued item or new input
            if not user_queue.empty():
                print(
                    f"[System] Processing next item in queue ({user_queue.qsize()} remaining)..."
                )


if __name__ == "__main__":
    # Start the input thread
    input_thread = threading.Thread(target=input_thread_func, daemon=True)
    input_thread.start()

    # Run the processing loop in the main thread
    process_queue()
