# main_agent.py

import os
import torch
from langchain.llms import OpenAI  # or from langchain.llms import HuggingFacePipeline
from langchain.agents import Tool, AgentType, initialize_agent

# Import your agents as Python classes
from agents.agent_script_writer import ScriptWriter
from agents.agent_text_to_speech import TextToSpeech
from agents.agent_text_to_image import TextToImage
from agents.agent_image_to_image import ImageToImage
from agents.agent_text_to_video import TextToVideo

# 1. Wrap each agent in a simple function that LangChain can call:

def script_writer_run(query: str) -> str:
    """Generates a short script from the query."""
    writer = ScriptWriter()
    return writer.generate_script(query)

def tts_run(query: str) -> str:
    """
    query: text to be spoken
    returns path to WAV file
    """
    tts = TextToSpeech()
    return tts.generate_speech(query)

def text2img_run(query: str) -> str:
    """
    query: e.g. "A photo-realistic sushi roll on a plate, 4K"
    returns path to generated image
    """
    t2i = TextToImage()
    return t2i.generate_image(query)

def img2img_run(query: str) -> str:
    """
    Example of query parsing: "prompt: 'Van Gogh style', image_path='my_image.png'"
    We'll do a simple parse for demonstration.
    """
    # parse prompt from query (super naive)
    # In production, you'd want a structured approach or a small function
    prompt = "Van Gogh style"
    image_path = "input.png"
    if "prompt:" in query:
        # attempt to parse
        import re
        match = re.search(r"prompt:\s*'(.*?)'", query)
        if match:
            prompt = match.group(1)
    if "image_path=" in query:
        match2 = re.search(r"image_path\s*=\s*'([^']+)'", query)
        if match2:
            image_path = match2.group(1)

    i2i = ImageToImage()
    return i2i.generate_image(prompt, image_path)

def text2video_run(query: str) -> str:
    """
    query: e.g. "A friendly chef making sushi in a bright kitchen"
    returns path to mp4
    """
    t2v = TextToVideo()
    return t2v.generate_video(query)

# 2. Create LangChain Tools

tool_script_writer = Tool(
    name="ScriptWriter",
    func=script_writer_run,
    description="Use this to write short scripts or commercials from user instructions."
)

tool_tts = Tool(
    name="TextToSpeech",
    func=tts_run,
    description="Use this to convert text to spoken audio (WAV)."
)

tool_text2img = Tool(
    name="TextToImage",
    func=text2img_run,
    description="Use this to generate images from text prompts."
)

tool_img2img = Tool(
    name="ImageToImage",
    func=img2img_run,
    description="Use this to transform an existing image with a text prompt. Query must contain 'prompt:' and 'image_path='"
)

tool_text2vid = Tool(
    name="TextToVideo",
    func=text2video_run,
    description="Use this to create a short video from text prompts."
)

tools = [tool_script_writer, tool_tts, tool_text2img, tool_img2img, tool_text2vid]

# 3. LLM for the "Agent Brain"
# Option A: Using OpenAI (requires OPENAI_API_KEY in your env).
# Option B: Use a local HF model with pipeline. Example:
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.llms import HuggingFacePipeline

def build_agent():
    """
    Builds a Zero-Shot agent that can parse user's natural language
    and pick the right tool(s).
    """
    # If you have an OpenAI key:
    llm = OpenAI(temperature=0)

    # If you want local:
    # model_id = "gpt2"  # or a bigger instruct model
    # local_pipe = pipeline("text-generation", model=model_id, max_length=512)
    # llm = HuggingFacePipeline(pipeline=local_pipe)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

if __name__ == "__main__":
    # Quick test
    agent = build_agent()
    print("Agent is ready!")
    response = agent.run("Write a short script about sushi, then transform this image with 'prompt:'Van Gogh style', image_path='test.png'")
    print("Final agent response:", response)
