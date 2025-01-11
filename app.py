# app.py
import gradio as gr
from agents.agent_script_writer import ScriptWriter
from agents.agent_text_to_speech import TextToSpeech
from agents.agent_text_to_image import TextToImage
from agents.agent_image_to_image import ImageToImage
from agents.agent_text_to_video import TextToVideo

# Instantiate each tool
script_writer = ScriptWriter()
tts = TextToSpeech()
t2i = TextToImage()
i2i = ImageToImage()
t2v = TextToVideo()

def generate_script(prompt):
    return script_writer.generate_script(prompt)

def generate_speech(script):
    return tts.generate_speech(script)

def generate_text2image(prompt):
    return t2i.generate_image(prompt)

def generate_image2image(prompt, init_image):
    init_path = "temp_input.png"
    init_image.save(init_path)
    return i2i.generate_image(prompt, init_path)

def generate_text2video(prompt):
    return t2v.generate_video(prompt)

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Modal AI Demo")

    with gr.Tab("Script Writer"):
        prompt_in = gr.Textbox(label="Script Prompt")
        script_out = gr.Textbox(label="Generated Script")
        generate_btn = gr.Button("Generate Script")
        generate_btn.click(fn=generate_script, inputs=prompt_in, outputs=script_out)

    with gr.Tab("Text to Speech"):
        script_in_tts = gr.Textbox(label="Text to speak")
        audio_out = gr.Audio(label="TTS Output")
        tts_btn = gr.Button("Generate Speech")
        tts_btn.click(fn=generate_speech, inputs=script_in_tts, outputs=audio_out)

    with gr.Tab("Text to Image"):
        prompt_img_in = gr.Textbox(label="Text prompt for an image")
        image_out = gr.Image(label="Generated Image")
        t2i_btn = gr.Button("Generate Image")
        t2i_btn.click(fn=generate_text2image, inputs=prompt_img_in, outputs=image_out)

    with gr.Tab("Image to Image"):
        prompt_img2img_in = gr.Textbox(label="Prompt for transformation")
        init_image_in = gr.Image(label="Upload initial image", type="pil")
        img2img_out = gr.Image(label="Transformed image")
        img2img_btn = gr.Button("Transform Image")
        img2img_btn.click(fn=generate_image2image, inputs=[prompt_img2img_in, init_image_in], outputs=img2img_out)

    with gr.Tab("Text to Video"):
        prompt_vid_in = gr.Textbox(label="Text prompt for short video")
        video_out = gr.Video(label="Generated Video")
        t2v_btn = gr.Button("Generate Video")
        t2v_btn.click(fn=generate_text2video, inputs=prompt_vid_in, outputs=video_out)

demo.launch()
