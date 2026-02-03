import gradio as gr
import os
from databricks import sql
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Databricks configuration from environment variables
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "your-genie-space-id")  # Add to secrets

def query_genie(user_question):
    """
    Query Databricks Genie and return the response
    """
    try:
        # Connect to Databricks
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=f"/sql/1.0/warehouses/{DATABRICKS_WAREHOUSE_ID}",
            access_token=DATABRICKS_TOKEN
        )
        
        cursor = connection.cursor()
        
        # Use Genie API or direct SQL query
        # For demo, using direct SQL - adapt based on your Genie setup
        logger.info(f"Executing query: {user_question}")
        
        # This is a simplified example - adapt to your Genie API integration
        cursor.execute(f"SELECT ai_query('{user_question}')")
        
        result = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        # Format response
        if result:
            response = "\n".join([str(row) for row in result])
            return response
        else:
            return "No results found."
            
    except Exception as e:
        logger.error(f"Error querying Genie: {str(e)}")
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Databricks Genie Assistant") as demo:
    gr.Markdown("# 🤖 Databricks Genie Assistant")
    gr.Markdown("Ask questions about your sales data in natural language!")
    
    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What were the top 5 products by revenue last month?",
                lines=3
            )
            submit_btn = gr.Button("Ask Genie", variant="primary")
        
        with gr.Column(scale=3):
            output = gr.Textbox(
                label="Genie's Response",
                lines=10,
                interactive=False
            )
    
    # Example questions
    gr.Markdown("### Example Questions:")
    gr.Examples(
        examples=[
            "What are the top 5 products by revenue?",
            "Show me sales trends by month",
            "Which store has the highest sales?",
            "What's our total revenue for Q4?",
        ],
        inputs=user_input
    )
    
    # Connect button to query function
    submit_btn.click(
        fn=query_genie,
        inputs=user_input,
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )