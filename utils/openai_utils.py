from openai import OpenAI

client = OpenAI()

def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Send prompt to OpenAI and return response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
