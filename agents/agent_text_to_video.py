# agents/agent_text_to_video.py
import uuid
import torch
from diffusers import DiffusionPipeline
from moviepy.editor import ImageSequenceClip

class TextToVideo:
    def __init__(self, model_id="damo-vilab/modelscope-text-to-video-synthesis"):
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")

    def generate_video(self, prompt: str, num_frames=16, num_inference_steps=50):
        """
        Create a short video from text prompt.
        """
        result = self.pipe(prompt, num_frames=num_frames, num_inference_steps=num_inference_steps)
        frames = result.frames  # list of PIL images

        filename = f"txt2vid_{uuid.uuid4().hex}.mp4"
        clip = ImageSequenceClip([frame for frame in frames], fps=8)
        clip.write_videofile(filename, fps=8, logger=None)
        return filename
