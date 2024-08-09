"""
Output Parser


"""

from langchain_core.output_parsers import StrOutputParser   
from langchain_core.messages import AIMessage

# typical AI respond
message = AIMessage(content="Hi Jim. How can I assist you today?")

parser = StrOutputParser()

output = parser.invoke(message)

print("Output: ", output)