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
        
        # DEBUG: show pending task and current url_map
        print(f"DEBUG Executor: pending_task={task}")
        print(f"DEBUG Executor: state.url_map={state.get('url_map')}")

        if not task:
            print("⚠️ Executor called with no pending task")
            return {"executor_memory": state.get("executor_memory", [])}
        
        print(f"\n--- Executor: Running action ---")
        print(f"Action: {task.get('action')}")
        
        # Resolve url_index => url if present (use state's url_map)
        if task and task.get('action') in ['open_website', 'open', 'open_app']:
            if task.get('url_index') is not None:
                url_map = state.get('url_map', {}) or {}
                print(f"DEBUG Executor: attempting to resolve url_index {task.get('url_index')} using url_map={url_map}")
                idx = task.get('url_index')
                # keys in url_map may be strings if serialized; try both
                mapped = url_map.get(idx)
                if mapped is None:
                    mapped = url_map.get(str(idx))
                if mapped:
                    # inject resolved url so handlers receive `url`
                    task['url'] = mapped
                    print(f"DEBUG Executor: resolved url_index {idx} -> {mapped}")
                else:
                    print(f"DEBUG Executor: could not resolve url_index {idx}; url_map keys: {list(url_map.keys())}")

        # Run the single task
        # Prevent duplicate opens by checking last_opened_url in state
        action_type = task.get('action')
        result = None
        last_opened = state.get('last_opened_url')
        if action_type in ['open_website', 'open_app', 'open']:
            # attempt to normalize URL if present
            url_candidate = task.get('url') or task.get('website') or task.get('destination')
            normalized = None
            if url_candidate:
                normalized = self._normalize_url(url_candidate)
            # If URL matches last opened, skip executing
            if normalized and last_opened and normalized == last_opened:
                print(f"ℹ️ Skipping duplicate open for {normalized}")
                result = [{"action": action_type, "result": f"Already opened: {normalized}"}]
            else:
                result = self.dispatch_actions([task])
        else:
            result = self.dispatch_actions([task])
        
        # CRITICAL: Keep thread alive for OS process handoff
        if action_type in ['open_website', 'open_app', 'open']:
            print("⏳ Waiting for process handoff...")
            time.sleep(3)  # Increased from 2.5 to 3 seconds
        
        # Append to memory
        existing_memory = state.get("executor_memory", [])
        
        print(f"Result: {result[0].get('result', 'No result')[:100]}...")
        
        # Generate completion message for streaming
        completion_messages = {
            "search": "✅ Search complete\n",
            "open_website": "✅ Website opened\n",
            "open_app": "✅ Application launched\n",
            "take_screenshot": "✅ Screenshot saved\n"
        }
        
        completion_msg = completion_messages.get(action_type, "✅ Action complete")
        
        # Update last_opened_url when we actually opened a URL
        updated_fields = {
            "executor_memory": existing_memory + result,
            "pending_task": None,  # Clear the task
            "reception_output": completion_msg  # ← Stream completion update
        }

        # If this was a search, try to extract URL map from the textual result
        try:
            if action_type == 'search' and result and isinstance(result, list):
                raw = result[0].get('result', '')
                # look for [URL_MAP]: block
                if '[URL_MAP]:' in raw:
                    print("DEBUG Executor: found [URL_MAP] in search result; parsing...")
                    # extract lines after the tag
                    start = raw.index('[URL_MAP]:') + len('[URL_MAP]:')
                    snippet = raw[start:]
                    # find all lines like [0]: https://...
                    url_map = {}
                    for line in snippet.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        # pattern: [i]: url
                        if line.startswith('[') and ']: ' in line:
                            try:
                                i_part, u = line.split(']: ', 1)
                                i = int(i_part.strip().lstrip('['))
                                url_map[i] = u.strip()
                            except Exception:
                                continue
                    if url_map:
                        updated_fields['url_map'] = url_map
                        print(f"DEBUG Executor: parsed url_map={url_map}")
                    else:
                        print("DEBUG Executor: found [URL_MAP] but parsed no entries")
        except Exception as e:
            print(f"⚠️ URL map extraction error in executor: {e}")

        if action_type in ['open_website', 'open_app', 'open']:
            # If we executed an actual open, capture the URL from task
            url_candidate = task.get('url') or task.get('website') or task.get('destination')
            if url_candidate:
                normalized = self._normalize_url(url_candidate)
                # If we skipped due to duplicate, last_opened remains same; otherwise set to normalized
                if not (result and isinstance(result, list) and result[0].get('result', '').startswith('Already opened')):
                    updated_fields['last_opened_url'] = normalized

        return updated_fields
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
            if not url:
                return "No URL provided."

            # Normalize URL and attempt to open in default browser
            normalized = self._normalize_url(url)
            try:
                # Prefer webbrowser.open which returns True/False
                try:
                    opened = webbrowser.open(normalized, new=2)
                    print(f"DEBUG Executor: webbrowser.open returned: {opened} for {normalized}")
                    if opened:
                        return normalized
                except Exception as e:
                    print(f"DEBUG Executor: webbrowser.open raised: {e}")

                # Fallback to Windows start command
                rc = os.system(f'start "" "{normalized}"')
                print(f"DEBUG Executor: os.system start rc={rc} for {normalized}")
                if rc == 0:
                    return normalized
                return f"Attempted to open {normalized} (rc={rc})"
            except Exception as e:
                return f"Failed to open website: {e}"

    def _normalize_url(self, url: str) -> str:
            if not url:
                return url
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                return 'https://' + url
            return url

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