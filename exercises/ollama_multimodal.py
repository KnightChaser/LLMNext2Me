# ollama_multimodal.py
# Working with Ollama multimodal LLMs

import base64
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def convert_to_base64(pil_image):
    """
    Converts a PIL Image to a Base64 encoded string.

    :param pil_image: PIL Image object.
    :return: Base64 encoded string of the image.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # Change format if necessary
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def prompt_function(data):
    """
    Constructs the prompt with separate image and text parts.

    :param data: Dictionary containing 'text' and 'image' keys.
    :return: List containing a HumanMessage with structured content.
    """
    text = data["text"]    # Extract text from data
    image = data["image"]  # Extract image from data

    # Define the image part
    image_part = {
        "type": "image_url",                             # Specify the file type
        "image_url": f"data:image/jpeg;base64,{image}",  # Base64 encoded image
    }

    # Define the text part
    text_part = {
        "type": "text",       # Specify that this part is text
        "text": text          # The actual text content
    }

    # Combine both parts into a single content list
    content_parts = [image_part, text_part]

    # Return the HumanMessage with the structured content
    return [HumanMessage(content=content_parts)]

def main():
    # Path to the image
    image_path = "/home/knightchaser/Pictures/Downloaded wallpaper/Flight 1.jpg"

    # Load the image
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Convert image to Base64
    image_b64 = convert_to_base64(pil_image)

    # Initialize the ChatOllama model with the llava variant
    llm = ChatOllama(model="llava:7b", temperature=0.7)

    # Establish the multimodal processing chain and invoke them
    chain = prompt_function | llm | StrOutputParser()
    response = chain.invoke(
        {"text": "Describe the given picture in bullet points.",
         "image": image_b64}
    )

    print(response)


if __name__ == "__main__":
    main()

