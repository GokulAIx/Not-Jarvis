import requests
import json
import uuid

# Hardcoded session ID for consistent testing
# WE CAN USE UUIDS OR ANY STRING IDENTIFIER
SESSION_ID = "UserX_session_1234"

def chat_with_jarvis(user_input):
    url = "http://localhost:8000/not-jarvis/stream"
    # Use the same thread_id for the entire conversation
    payload = {
        "user_goal": user_input,
        "thread_id": SESSION_ID
    }

    try:
        # Use stream=True to handle the StreamingResponse from FastAPI
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            for line in response.iter_lines():
                if line:
                    # Decode the 'data: ' prefix from the event stream
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        content = decoded_line.replace("data: ", "")
                        
                        # Check for end signal
                        if content == "[DONE]":
                            break
                        
                        # Handle potential JSON strings within the text
                        try:
                            # If the content is a JSON-stringified object, clean it
                            data = json.loads(content)
                            if isinstance(data, list) and len(data) > 0:
                                # Extract text if it's a list of blocks
                                print(data[0].get('text', ''))
                            else:
                                print(data)
                        except json.JSONDecodeError:
                            # If it's just raw text, print it directly (respects \n in the string)
                            print(content)
            print() # Final newline after response ends
    except Exception as e:
        print(f"\n[Client Error]: {e}")

if __name__ == "__main__":
    print("Not Jarvis Terminal Client (type 'exit' to stop, 'reset' for new session)")
    print(f"Session ID: {SESSION_ID}\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        if user_input.lower() == 'reset':
            SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"
            print(f"🔄 New session started: {SESSION_ID}\n")
            continue
        print("--- Sending Request ---")
        chat_with_jarvis(user_input)