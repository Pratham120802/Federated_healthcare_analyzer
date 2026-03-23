import requests


def generate_insight(accuracy, loss):
    prompt = f"""
You are assisting with a healthcare federated learning project.

Model results:
- Accuracy: {accuracy:.4f}
- Loss: {loss:.4f}

Write a short, simple interpretation for a professor or clinician.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response returned by local LLM.")
    except Exception as e:
        return f"LLM insight unavailable: {e}"