import requests
import gradio as gr

url = "http://ml-model-service"

def predict_value(x):
    try:
        response = requests.get(f"{url}/predict?x={x}")
        
        if response.status_code == 200:
            result = response.json()
            return f"Predicted Y: {result['predicted_y']}"
        else:
            return f"Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"
    
iface = gr.Interface(
    fn=predict_value,
    inputs=gr.Number(label="Input X", value=5.0),
    outputs="text",
    title="Linear Regression Model",
    description="Enter a number to get the prediction (y = 2x + 1)",
    examples=[[1], [2], [3], [4], [5], [10], [20]]
)

# Launch the app
iface.launch(server_name="0.0.0.0", server_port=7860)
