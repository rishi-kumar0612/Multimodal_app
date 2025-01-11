# agents/agent_script_writer.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class ScriptWriter:
    def __init__(self, model_id="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_script(self, prompt: str, max_length=100):
        """
        Generate a short script (text) from a prompt.
        """
        output = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return output[0]["generated_text"]

