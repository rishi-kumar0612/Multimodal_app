# agents/agent_text_to_image.py
import uuid
import torch
from diffusers import StableDiffusionPipeline

class TextToImage:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        # Load pipeline once
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")  # or "cpu" if no GPU

    def generate_image(self, prompt: str):
        """
        Generate an image from text and save locally.
        """
        image = self.pipe(prompt).images[0]
        filename = f"txt2img_{uuid.uuid4().hex}.png"
        image.save(filename)
        return filename
