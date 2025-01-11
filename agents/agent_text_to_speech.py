# agents/agent_text_to_speech.py
import uuid
from TTS.api import TTS

class TextToSpeech:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        # Downloads the TTS model on init
        self.tts = TTS(model_name)

    def generate_speech(self, text: str):
        """
        Convert text to speech (WAV file) and return the filename.
        """
        filename = f"tts_{uuid.uuid4().hex}.wav"
        self.tts.tts_to_file(text=text, file_path=filename)
        return filename
