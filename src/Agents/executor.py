import subprocess
import os
import inspect
import webbrowser
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
        dispatch_results = self.dispatch_actions([step.dict() for step in steps])
        state["executor_output"] = dispatch_results
        return state


    def dispatch_actions(self, steps):
        results=[]
        action_handlers = {
        "open_app": self.open_app,
        "open_website": self.open_website,
        "take_screenshot": self.take_screenshot,
        # Add more mappings as you add features
    }
        for step in steps:
            action_type = step.get("action")
            handler = action_handlers.get(action_type)
            if handler:
                sig = inspect.signature(handler)
                params = {k: v for k, v in step.items() if k != "action" and k in sig.parameters and v is not None}
                result = handler(**params)
                results.append({
                    "action": action_type,
                    "action":params,
                    "result": result}
                )

            else:
                results.append({
                "action": action_type,
                "params": {},
                "result": f"No handler for action: {action_type}"
            })
        return results


    def open_app(self, app_name: str) -> str:
        if not app_name or not isinstance(app_name, str):
            return "Invalid application name."

        if self.allowed_apps and app_name.lower() not in [a.lower() for a in self.allowed_apps]:
            return f"App '{app_name}' is not allowed."

        try:
            subprocess.Popen(["start", "", app_name], shell=True)
            return f"Opening '{app_name}'..."
        except Exception as e:
            return f"Failed to open '{app_name}': {e}"


    def open_website(self, url: str) -> str:
        if not url or not isinstance(url, str):
            return "Invalid URL."
        try:
            webbrowser.open(url)
            return f"Opening website: {url}"
        except Exception as e:
            return f"Failed to open website: {e}"

    def take_screenshot(self, save_path: str = None) -> str:
        if ImageGrab is None:
            return "Pillow (PIL) is not installed. Cannot take screenshot."
        if not save_path:
            save_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        try:
            img = ImageGrab.grab()
            img.save(save_path)
            return f"Screenshot saved to {save_path}"
        except Exception as e:
            return f"Failed to take screenshot: {e}"
        

# tr=Executor(allowed_apps=["CHROME","NOTEPAD"])
# tr.open_app("CHROME")

