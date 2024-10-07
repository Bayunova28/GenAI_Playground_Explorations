## Text-To-Text Generation
<img align="center" src="https://github.com/Bayunova28/GenAI_Playground_Explorations/blob/main/assets/gemini-architecture.jpg" height="450" width="1000">

Google Gemini AI is a powerful LLM model that can generate high-quality text and images for various use cases. It is currently available for free for anyone who wants to try it out. Gemini AI has two models: Gemini-Pro and Gemini-Pro-Vision. Gemini-Pro is a good recommendation for text-based use cases, such as writing blog posts, summaries, captions, etc. Gemini-Pro-Vision is designed for image-based use cases, such as captioning, describing, storytelling, and more. Both versions use state-of-the-art neural networks and large-scale datasets to produce coherent and relevant texts. If you are looking for a tool to enhance your text or image projects, you should check out Google Gemini AI today in [here](https://aistudio.google.com/app/apikey).

## Text-To-Image Generation
<img align="center" src="https://github.com/Bayunova28/GenAI_Playground_Explorations/blob/main/assets/rabbit-generative-ai.png" height="350" width="1000">

<p align="justify">Stable Diffusion is based on a type of diffusion model that is called Latent Diffusion, which details can be seen in the paper High-Resolution Image Synthesis with Latent Diffusion Models. These diffusion models have gained popularity in recent years, specially for their ability to achieve state-of-the-art results in generating image data. However, diffusion models can consume a lot of memory and be computationally expensive to work with. Stable Diffusion is a powerful machine learning model that enables users to generate high-quality images, making it a valuable tool for artists, designers, and researchers. In this guide, we’ll walk you through the steps to set up a Stable Diffusion model on your machine.<p>

### Generate the Prompt 
```python
# Install Library
!pip install diffusers

# Import Library
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

# Define function to generate 3 RGB image
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# Load the pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# If you have a GPU, use it
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
# Define a new text prompt for a nature view
num_images = 3
prompt = ["Cute Rabbit, Ultra HD, realistic, futuristic, sharp, octane render, photoshopped, photorealistic, soft, pastel, Aesthetic, Magical background"] * num_images
images = pipe(prompt).images

# Create a grid of the generated images
grid = image_grid(images, rows=1, cols=3)
grid
```
## Image-To-Text Generation
<div align="center">
	<img width = "90%" src="https://github.com/Bayunova28/GenAI_Playground_Explorations/blob/main/assets/women-generative-ai.png">
  <img width = "40%" src="https://github.com/Bayunova28/GenAI_Playground_Explorations/blob/main/assets/women-gun-generative-ai.png">
  <img width = "48%" src="https://github.com/Bayunova28/GenAI_Playground_Explorations/blob/main/assets/spongebob-generative-ai.png">
</div>
<p align="justify">Salesforce’s BLIP model is designed to seamlessly integrate vision and language tasks, making it an ideal choice for image captioning. By leveraging extensive pre-training, BLIP can generate high-quality captions that accurately describe images, opening up a myriad of possibilities for applications. BLIP, which stands for Bootstrapping Language-Image Pre-training, is like a highly advanced AI student who has mastered the art of understanding and creating content involving both images and text. But what sets BLIP apart is its innovative learning approach and the remarkable range of tasks it can perform. In this guide, we’ll walk you through the steps to set up a Stable Diffusion model on your machine.<p>

### Generate the Prompt 
```python
# Install library
!pip install transformers

# Import library
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate text from image
def generate_caption(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs)

    # Decode and return the caption
    caption = processor.decode(output[0], skip_special_tokens=True)
    return image, caption.capitalize()

# Example usage
image_path = "YOUR-PATH-IMAGE"  # Change to your image path
image, caption = generate_caption(image_path)
# Display the image and the generated caption
plt.imshow(image)
plt.axis('off')
plt.title(caption)
plt.show()
```
## Acknowledgement
* [Gemini: The Power of the Most Capable AI Model by Google](https://medium.com/@pankaj_pandey/gemini-the-power-of-the-most-capable-ai-model-by-google-ff72ddc66c2d)
* [Exploring Stable Diffusion for Image Generation](https://medium.com/@souvik_real/exploring-stable-diffusion-for-image-generation-generating-code-in-python-7c1e56371b78)
* [Building an Image Captioning Model Using Salesforce’s BLIP Model](https://medium.com/@k.pranav_22/building-an-image-captioning-model-using-salesforces-blip-model-3b80a4f032c4)
