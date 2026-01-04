import subprocess
import os
import inspect
import webbrowser
import time  # Essential for the delay
from datetime import datetime

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

class Executor:
    def __init__(self, allowed_apps=None):
        self.allowed_apps = allowed_apps or []

    def what_execute(self, state):
        task = state.get("pending_task")
        
        if not task:
            print("⚠️ Executor called with no pending task")
            return {"executor_memory": state.get("executor_memory", [])}
        
        print(f"\n--- Executor: Running action ---")
        print(f"Action: {task.get('action')}")
        
        # Run the single task
        result = self.dispatch_actions([task])
        
        # CRITICAL: Keep thread alive for OS process handoff
        action_type = task.get('action')
        if action_type in ['open_website', 'open_app', 'open']:
            print("⏳ Waiting for process handoff...")
            time.sleep(3)  # Increased from 2.5 to 3 seconds
        
        # Append to memory
        existing_memory = state.get("executor_memory", [])
        
        print(f"Result: {result[0].get('result', 'No result')[:100]}...")
        
        return {
            "executor_memory": existing_memory + result,
            "pending_task": None  # Clear the task
        }
    def dispatch_actions(self, steps):
        # Import tools locally to avoid circular imports
        from ..tools.tools import search_tool
        
        results = []
        action_handlers = {
            "open_app": self.open_app,
            "open_website": self.open_website,
            "open": self.open_website,
            "take_screenshot": self.take_screenshot,
            "search": lambda query: search_tool.run(query)
        }
        
        for step in steps:
            action_type = step.get("action")
            handler = action_handlers.get(action_type)
            
            if handler:
                sig = inspect.signature(handler)
                # Filter out 'action' key and only pass what the handler accepts
                params = {k: v for k, v in step.items() if k in sig.parameters and v is not None}
                
                try:
                    result = handler(**params)
                    results.append({"action": action_type, "result": result})
                except Exception as e:
                    results.append({"action": action_type, "result": f"Error: {str(e)}"})
            else:
                results.append({"action": action_type, "result": f"No handler found for {action_type}"})
        
        return results

    def open_website(self, url: str) -> str:
            if not url: return "No URL provided."
            
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            try:
                # Use Windows shell to open URL in default browser
                os.system(f'start "" "{url}"')
                return f"Opening website: {url}"
            except Exception as e:
                return f"Failed to open website: {e}"

    def open_app(self, app_name: str) -> str:
        try:
            # 'start' is a Windows command to launch apps/files
            subprocess.Popen(["start", "", app_name], shell=True)
            return f"Opening {app_name}..."
        except Exception as e:
            return f"Failed: {e}"

    def take_screenshot(self, save_path: str = None) -> str:
        if not ImageGrab: return "Pillow not installed."
        save_path = save_path or f"screenshot_{datetime.now().strftime('%H%M%S')}.png"
        img = ImageGrab.grab()
        img.save(save_path)
        return f"Saved to {save_path}"