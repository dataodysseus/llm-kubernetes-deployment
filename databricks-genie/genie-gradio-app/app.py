import gradio as gr
import os
import requests
import logging
from typing import Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Databricks configuration from environment variables
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")

# Validate required environment variables
required_vars = {
    "DATABRICKS_HOST": DATABRICKS_HOST,
    "DATABRICKS_TOKEN": DATABRICKS_TOKEN,
    "GENIE_SPACE_ID": GENIE_SPACE_ID
}

missing_vars = [k for k, v in required_vars.items() if not v]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info(f"Connected to Databricks workspace: {DATABRICKS_HOST}")
logger.info(f"Using Genie Space: {GENIE_SPACE_ID}")


def query_genie(user_question: str) -> str:
    """
    Query Databricks Genie using the REST API
    
    Args:
        user_question: Natural language question from user
        
    Returns:
        Formatted response string
    """
    if not user_question or not user_question.strip():
        return "⚠️ Please enter a question."
    
    try:
        logger.info(f"User query: {user_question}")
        
        # Databricks Genie API endpoint
        url = f"https://{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation"
        
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Create conversation and ask question
        payload = {
            "content": user_question
        }
        
        logger.info(f"Calling Genie API: {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Genie API response received")
            
            # Extract the conversation ID
            conversation_id = result.get("conversation_id")
            message_id = result.get("message_id")
            
            if not conversation_id or not message_id:
                return "❌ Error: Invalid response from Genie API"
            
            # Poll for the result
            result_url = f"https://{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conversation_id}/messages/{message_id}"
            
            # Wait for response (poll up to 30 seconds)
            import time
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(1)
                
                poll_response = requests.get(result_url, headers=headers, timeout=10)
                
                if poll_response.status_code == 200:
                    poll_result = poll_response.json()
                    status = poll_result.get("status")
                    
                    if status == "COMPLETED":
                        # Extract the answer
                        attachments = poll_result.get("attachments", [])
                        
                        # Look for text response
                        for attachment in attachments:
                            if attachment.get("text"):
                                text_content = attachment["text"].get("content", "")
                                if text_content:
                                    logger.info(f"Query successful")
                                    return text_content
                        
                        # Look for query results
                        query_result = poll_result.get("query_result")
                        if query_result:
                            # Format the result
                            data = query_result.get("data_typed_array", [])
                            if data:
                                # Format as table
                                return format_query_result(data)
                        
                        return "✅ Query completed but no results found."
                    
                    elif status == "FAILED":
                        error_msg = poll_result.get("error", {}).get("message", "Unknown error")
                        logger.error(f"Genie query failed: {error_msg}")
                        return f"❌ Query failed: {error_msg}"
            
            return "⏱️ Query timed out. Please try again with a simpler question."
        
        elif response.status_code == 401:
            return "❌ Authentication failed. Please check your Databricks token."
        
        elif response.status_code == 404:
            return f"❌ Genie Space not found. Please verify Space ID: {GENIE_SPACE_ID}"
        
        else:
            error_text = response.text
            logger.error(f"API error {response.status_code}: {error_text}")
            return f"❌ API Error ({response.status_code}): {error_text}"
            
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        return "⏱️ Request timeout. Please try again."
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        return "❌ Cannot connect to Databricks. Please check your network."
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"❌ Error: {str(e)}"


def format_query_result(data: list) -> str:
    """
    Format query result data into readable text
    
    Args:
        data: List of rows from query result
        
    Returns:
        Formatted string representation
    """
    if not data:
        return "No results found."
    
    # Get column names from first row
    if len(data) > 0 and isinstance(data[0], dict):
        columns = list(data[0].keys())
        
        # Create header
        result = "| " + " | ".join(columns) + " |\n"
        result += "|-" + "-|-".join(["-" * len(col) for col in columns]) + "-|\n"
        
        # Add rows
        for row in data[:10]:  # Limit to 10 rows for display
            values = [str(row.get(col, "")) for col in columns]
            result += "| " + " | ".join(values) + " |\n"
        
        if len(data) > 10:
            result += f"\n... and {len(data) - 10} more rows"
        
        return result
    
    # Fallback to simple format
    return json.dumps(data, indent=2)


# Custom CSS for better styling
custom_css = """
#component-0 {
    max-width: 1400px;
    margin: auto;
    padding: 20px;
}
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Databricks Genie Assistant", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🤖 Databricks Genie Assistant
        ### Ask questions about your sales data in natural language!
        
        This AI assistant is powered by Databricks Genie and has access to your complete sales database.
        """
    )
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Your Question")
            user_input = gr.Textbox(
                label="",
                placeholder="e.g., What are the top 5 products by revenue?",
                lines=4,
                max_lines=10
            )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Ask Genie", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Genie's Response")
            output = gr.Textbox(
                label="",
                lines=12,
                max_lines=25,
                interactive=False,
                show_copy_button=True
            )
    
    # Example questions
    with gr.Accordion("📝 Example Questions", open=False):
        gr.Examples(
            examples=[
                ["What are the top 5 products by revenue?"],
                ["Show me sales trends by month"],
                ["Which store has the highest sales?"],
                ["What's our total revenue for Q4?"],
            ],
            inputs=user_input,
            label="Click any example to try it:"
        )
    
    # Connection status
    with gr.Accordion("🔗 Connection Info", open=False):
        gr.Markdown(
            f"""
            **Databricks Workspace:** `{DATABRICKS_HOST}`  
            **Genie Space ID:** `{GENIE_SPACE_ID}`
            
            Status: ✅ Connected
            """
        )
    
    # Event handlers
    submit_btn.click(
        fn=query_genie,
        inputs=user_input,
        outputs=output,
        api_name="query"
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[user_input, output]
    )
    
    # Allow Enter key to submit
    user_input.submit(
        fn=query_genie,
        inputs=user_input,
        outputs=output
    )


# Launch the app
if __name__ == "__main__":
    logger.info("Starting Databricks Genie Gradio App...")
    logger.info(f"Databricks Host: {DATABRICKS_HOST}")
    logger.info(f"Genie Space ID: {GENIE_SPACE_ID}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )