# app/session_manager.py

from typing import Dict

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def create_session(self, user_name: str):
        """Creates a new session for a user."""
        self.sessions[user_name] = {"history": []}

    def add_to_history(self, user_name: str, message: str):
        """Adds a message to the user's session history."""
        if user_name in self.sessions:
            self.sessions[user_name]["history"].append(message)

    def get_history(self, user_name: str):
        """Retrieves the session history for a user."""
        return self.sessions.get(user_name, {}).get("history", [])
