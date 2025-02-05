import os
import logging
import torch
from PIL import Image
import gradio as gr

from deepseek_vl2.serve.app_modules import gradio_utils, overwrites, presets, utils
from deepseek_vl2.serve.inference import convert_conversation_to_prompts, deepseek_generate, load_model

# Setting up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Helper function for loading images
def load_images(images):
    pil_images = []
    for img_or_file in images:
        try:
            if isinstance(img_or_file, Image.Image):
                pil_images.append(img_or_file)
            else:
                image = Image.open(img_or_file.name).convert("RGB")
                pil_images.append(image)
        except IOError as e:
            logger.error(f"Error opening image: {e}")
            raise ValueError("Invalid image file provided.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ValueError("An unexpected error occurred while loading the image.")
    return pil_images

# Helper function to handle None values for text and images
def handle_none_values(text, images):
    if text is None:
        text = ""
    if images is None:
        images = []
    return text, images

# Helper function for formatting the conversation
def format_conversation(conversation, format_type="deepseek"):
    ret = ""
    system_prompt = conversation.system_template.format(system_message=conversation.system_message) if conversation.system_message else ""
    
    for i, (role, message) in enumerate(conversation.messages):
        if format_type == "deepseek":
            seps = [conversation.sep, conversation.sep2]
            ret += role + ": " + message + seps[i % 2]
        else:
            # Handle other formats if necessary
            pass
    return ret

# Main function to generate response with history and images
def generate_response(conversation, tokenizer, vl_gpt, vl_chat_processor, stop_words, max_length, temperature, top_p):
    full_response = ""
    with torch.no_grad():
        for x in deepseek_generate(
            conversations=conversation,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            chunk_size=conversation.chunk_size
        ):
            full_response += x
    return full_response

# Function to load and prepare models
def load_models(model_path, device='cuda'):
    logger.info("Loading models from path: %s", model_path)
    model = load_model(model_path, device)
    return model

# Function to handle and process the conversation prompt
def get_prompt(conversation, tokenizer, vl_chat_processor, stop_words, max_length=None, temperature=0.7, top_p=0.9):
    text, images = handle_none_values(conversation.text, conversation.images)
    
    # Prepare prompt
    prompt = format_conversation(conversation, format_type="deepseek")
    if max_length is None:
        max_length = 2048

    logger.debug("Generated prompt: %s", prompt)
    
    response = generate_response(
        conversation=conversation,
        tokenizer=tokenizer,
        vl_gpt=vl_chat_processor,
        vl_chat_processor=vl_chat_processor,
        stop_words=stop_words,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p
    )
    return response

# Function for resetting and managing the conversation state
def reset_state():
    logger.info("Resetting state.")
    # Reset or clear any necessary global variables or session states

# Main function for image and text input processing
def predict(images, text, conversation, vl_gpt, vl_chat_processor, tokenizer, stop_words, max_length=2048, temperature=0.7, top_p=0.9):
    try:
        logger.info("Starting prediction process.")
        
        # Step 1: Load and process images
        pil_images = load_images(images)
        
        # Step 2: Process the prompt and generate response
        response = get_prompt(conversation, tokenizer, vl_chat_processor, stop_words, max_length, temperature, top_p)
        
        # Step 3: Handle outputs and return result
        logger.info("Prediction complete.")
        return response
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}

# Function for running the Gradio interface
def launch_gradio_interface():
    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(label="Input Image(s)", type="pil", multiple=True),
            gr.Textbox(label="Input Text", lines=2)
        ],
        outputs=[
            gr.Textbox(label="Response"),
        ],
        title="DeepSeek Prediction",
        description="This application generates responses based on images and text input using the DeepSeek model."
    )
    interface.launch()

# Main entry point for the program
if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/model"
    vl_gpt = load_models(model_path)
    vl_chat_processor = "your_vl_chat_processor_here"
    tokenizer = "your_tokenizer_here"
    stop_words = ["stopword1", "stopword2"]  # Customize this list
    launch_gradio_interface()
