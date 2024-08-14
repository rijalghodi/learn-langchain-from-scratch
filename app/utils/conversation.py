from typing import Callable

class Conversation():
    def __init__(self, ask_with_session: Callable[[str, str], any], welcome_text: str = "Welcome to Bot App", 
                 human_alias: str = 'You', ai_alias: str = 'AI') -> None:
        self.ask_with_session = ask_with_session
        self.welcome_text = welcome_text
        self.human_alias = human_alias
        self.ai_alias = ai_alias
    
    def chat(self):
        print("\n\n")
        print(self.welcome_text)
        while True:
            session_id = input("\nEnter Session ID: ")
            if session_id.lower() == "exit":
                print("\n--- App ended ---")
                break
            # Chat loops
            while True:
                query = input(f"\n{self.human_alias}: ")
                if query.lower() == "exit":
                    print("\n--- Session ended ---")
                    break

                # Get AI response using history
                response = self.ask_with_session(session_id, query)

                print(f"\n{self.ai_alias}: {response}")