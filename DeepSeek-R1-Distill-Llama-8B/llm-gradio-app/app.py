import os
import requests
import gradio as gr

url = os.getenv("LLM_SERVICE_URL", "http://llm-service")

def generate_response(user_input):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond to each question accurately and concisely."
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    }

    response = requests.post(f"{url}/generate", json=payload)

    if response.status_code == 200:
        response_text = response.json()['response']
        assistant_start = response_text.find("<|assistant|>")
        if assistant_start != -1:
            cleaned_output = response_text[assistant_start + len("<|assistant|>"):].strip()
            return cleaned_output
        else:
            return "Could not find <|assistant|> in the response."
    else:
        return f"Error: {response.status_code} {response.text}"
    
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Chatbot",
    description="Ask me anything! "
)

# Launch the app
iface.launch(server_name="0.0.0.0", server_port=7860)