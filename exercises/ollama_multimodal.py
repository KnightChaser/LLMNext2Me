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
    text = data["text"]  # Extract text from data
    image = data["image"]  # Extract image from data

    # Define the image part
    image_part = {
        "type": "image_url",  # Specify the file type
        "image_url": f"data:image/jpeg;base64,{image}",  # Base64 encoded image
    }

    # Define the text part
    text_part = {
        "type": "text",  # Specify that this part is text
        "text": text,  # The actual text content
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
        {"text": "Describe the given picture in bullet points.", "image": image_b64}
    )

    print(response)


if __name__ == "__main__":
    main()

# - The image is a high-resolution, colorful digital graphic or artwork.
# - It depicts a scene with a vintage airplane flying low over a town or village.
# - The airplane has a propeller engine and an open cockpit, suggesting it's a small or light aircraft commonly used for recreational flying.
# - There is an individual piloting the aircraft, visible through the cockpit window.
# - Below the airplane, there are buildings with red roofs and white walls, indicative of a Mediterranean architectural style.
# - The town below has a variety of structures including houses, shops, and what appears to be a church or chapel.
# - The ground is covered in cobblestones, which adds to the quaint and old-world charm of the scene.
# - In the foreground, there are colorful buildings with a variety of designs and decorations, including umbrellas, which give the impression that this could be a tourist destination or perhaps a set from a themed attraction.
# - The overall style of the image is reminiscent of a video game or animated film due to its high level of detail and vibrant colors.
# - The sky above is clear, with no visible clouds, indicating good weather conditions for flying.
# - There are no texts or inscriptions visible in the image.
