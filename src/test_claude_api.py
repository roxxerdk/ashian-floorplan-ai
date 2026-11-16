import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY not found")

client = Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=100,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Claude API connection successful."
        }
    ]
)

print("Response:", response.content[0].text)
print("Input tokens:", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)
print("API connection verified.")
