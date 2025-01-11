# agents/agent_image_to_image.py
import uuid
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

class ImageToImage:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")

    def generate_image(self, prompt: str, init_image_path: str, strength=0.8, guidance=7.5):
        """
        Transform an existing image with a prompt.
        """
        init_image = Image.open(init_image_path).convert("RGB")
        init_image = init_image.resize((512, 512))

        result = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance
        )
        out_image = result.images[0]

        filename = f"img2img_{uuid.uuid4().hex}.png"
        out_image.save(filename)
        return filename
