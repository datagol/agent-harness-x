"""Simulated app state for the GroceryBuddy simple_chat demo.

A dict (not globals), so tests can build their own instance via
dataclasses.replace()-style copy before handing it to the tool factory.
"""

grocery_state = {
    "list": [],
    "memory": {},
    "staples": ["whole milk", "eggs", "bananas", "coffee", "sourdough bread"],
    "staples_status": "ok",  # 'ok' | 'none' | 'error'
    "fetch_mode": "live",  # 'live' | 'mock_fail'
}
