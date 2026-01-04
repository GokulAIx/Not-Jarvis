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
        steps = state.get("pending_tasks", [])
        print(f"DEBUG EXECUTOR: Received {len(steps)} tasks")
        
        # 1. Safely convert Pydantic objects to dicts
        # This is the most common reason for 'Page not opening'
        clean_steps = []
        for step in steps:
            if hasattr(step, "dict"):
                clean_steps.append(step.dict())
            else:
                clean_steps.append(step)

        # 2. Run the actions
        dispatch_results = self.dispatch_actions(clean_steps)
        
        # 3. Handshake Delay: Keep thread alive for the OS to catch the browser request
        if any(s.get("action") in ["open_app", "open_website"] for s in clean_steps):
            print("DEBUG: Waiting for OS process handoff...")
            time.sleep(2.5) 

        # 4. Clear tasks so they don't repeat
        return {
            "executor_output": dispatch_results,
            "pending_tasks": [] 
        }

    def dispatch_actions(self, steps):
        results = []
        action_handlers = {
            "open_app": self.open_app,
            "open_website": self.open_website,
            "open": self.open_website,  # Alias for LLM flexibility
            "take_screenshot": self.take_screenshot,
        }
        for step in steps:
            action_type = step.get("action")
            handler = action_handlers.get(action_type)
            if handler:
                sig = inspect.signature(handler)
                # Filter out 'action' key and only pass what the handler accepts
                params = {k: v for k, v in step.items() if k in sig.parameters and v is not None}
                result = handler(**params)
                results.append({"action": action_type, "result": result})
            else:
                results.append({"action": action_type, "result": f"No handler found for {action_type}"})
        return results

    def open_website(self, url: str) -> str:
            if not url: return "No URL provided."
            try:
                # Force Windows to handle the URL through the shell
                # This is often more reliable than webbrowser.open in async environments
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