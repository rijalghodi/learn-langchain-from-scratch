from typing import Callable

class Conversation():
    def __init__(self, ask_with_session: Callable[[str, str], any]) -> None:
        self.ask_with_session = ask_with_session
    
    def chat(self):
        while True:
            session_id = input("\nEnter Session ID: ")
            if session_id.lower() == "exit":
                print("\nApp ended")
                break
            # Chat loops
            while True:
                query = input("\nYou: ")
                if query.lower() == "exit":
                    print("\nSession ended")
                    break

                # Get AI response using history
                response = self.ask_with_session(session_id, query)

                print(f"\nAI: {response}")