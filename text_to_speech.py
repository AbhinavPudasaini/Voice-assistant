import os
import uuid
import queue
import threading
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

load_dotenv()

ELEVENLABS_API_KEY = "elleven_labs_api_key"
if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY is not set. Put it in your .env or environment variables.")
 
elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def stream_text_to_speech(text: str, voice_id: str = "56AoDkrOh6qfVPDXZ7Pt", interrupt_event: threading.Event = None):
    """
    Streams audio from ElevenLabs and plays it immediately.
    Checks interrupt_event to stop playback early.
    """
    print("Generating and streaming audio...")
    
    audio_stream = elevenlabs.text_to_speech.convert(
        voice_id=voice_id,
        output_format="pcm_16000",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )
    
    # Simpler approach: Blocking write to stream
    # We open the stream and write chunks as they arrive
    # with sd.OutputStream(samplerate=16000, channels=1, dtype='int16') as stream:
    #     for chunk in audio_stream:
    #         if chunk:
    #             # Convert bytes to numpy array
    #             data = np.frombuffer(chunk, dtype=np.int16)
    #             stream.write(data)

    with sd.OutputStream(samplerate=16000, channels=1, dtype='int16') as stream:
        for chunk in audio_stream:
            if interrupt_event and interrupt_event.is_set():
                return
            if chunk:
                data = np.frombuffer(chunk, dtype=np.int16)
                stream.write(data)

# Keep old functions for compatibility if needed, but we replace them in main.py
def text_to_speech_file(text: str, *, out_dir: str = ".", voice_id: str = "56AoDkrOh6qfVPDXZ7Pt") -> str:
    """
    Generates an MP3 using ElevenLabs and saves it to disk. Returns the MP3 path.
    """
    os.makedirs(out_dir, exist_ok=True)

    response = elevenlabs.text_to_speech.convert(
        voice_id=voice_id,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    save_file_path = os.path.join(out_dir, f"{uuid.uuid4()}.mp3")

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return save_file_path

def play_audio_file(path: str) -> None:
    """
    Plays an audio file on Windows using the default associated player (non-blocking).
    """
    # if not path or not os.path.exists(path):
    #     return
    os.startfile(os.path.abspath(path))