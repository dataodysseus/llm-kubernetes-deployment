import requests
import gradio as gr

# LLM API URL
url = "http://llm-service"

# Function to generate response while maintaining chat history
def generate_response(user_input, chat_history):
    # Start with system message
    messages = [{"role": "system", "content": "You are a friendly chatbot, respond to each question accurately and concisely."}]
    
    # Append previous conversation history
    messages.extend(chat_history)

    # Add new user message
    messages.append({"role": "user", "content": user_input})

    # Send request to LLM service
    response = requests.post(f"{url}/generate", json={"messages": messages})

    if response.status_code == 200:
        response_text = response.json().get('response', '')

        # Extract response after <|assistant|>
        assistant_start = response_text.find("<|assistant|>")
        if assistant_start != -1:
            cleaned_output = response_text[assistant_start + len("<|assistant|>"):].strip()
        else:
            cleaned_output = response_text  # Fallback to full response

        # Append assistant's response to chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": cleaned_output})

        return chat_history, chat_history
    else:
        return chat_history, f"Error: {response.status_code} {response.text}"

# Gradio interface with chat history support
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here..."),  # User input
        gr.State([])  # Chat history (persists across interactions)
    ],
    outputs=[
        gr.Chatbot(),  # Display chat history
        gr.State([])   # Maintain state of chat history
    ],
    title="Chatbot with Memory",
    description="A chatbot that remembers the conversation!"
)

# Launch the app
iface.launch(server_name="0.0.0.0", server_port=7860)