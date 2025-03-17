import requests
import gradio as gr

url = "http://llm-service"

def generate_response(user_input):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
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
        output = response_text[0]
        cleaned_output = output.split("<start_of_turn>model")[1].strip().split("<end_of_turn>")[0].strip()
        output_list = [line.strip() for line in cleaned_output.split("\n") if line.strip()]
        result = "\n".join(output_list)
        return result
    else:
        return f"Error: {response.status_code} {response.text}"
    
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Chatbot",
    description="Ask me anything!"
)

# Launch the app
iface.launch(server_name="0.0.0.0", server_port=7860)