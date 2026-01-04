import requests
import json

def chat_with_jarvis(user_input):
    url = "http://localhost:8000/not-jarvis/stream"
    payload = {
        "user_goal": user_input,
        "thread_id": "gokul_dev_session_001"
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
                                print(data[0].get('text', ''), end="", flush=True)
                            else:
                                print(data, end="", flush=True)
                        except json.JSONDecodeError:
                            # If it's just raw text, print it directly
                            print(content, end="", flush=True)
            print() # Final newline after response ends
    except Exception as e:
        print(f"\n[Client Error]: {e}")

if __name__ == "__main__":
    print("Not Jarvis Terminal Client (type 'exit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        print("--- Sending Request ---")
        chat_with_jarvis(user_input)