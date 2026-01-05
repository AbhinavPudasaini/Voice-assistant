import os
import json
import tempfile
import threading
import queue
import time
import sounddevice as sd
import numpy as np
from groq import Groq
import wave

# ...existing code...
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "groq_api_key")
client = Groq(api_key=GROQ_API_KEY)

samplerate = 16000
channels = 1
chunk_duration = 3  # seconds

def record_chunk_to_wav(duration=chunk_duration):
    """Record audio chunk from microphone and return temp WAV path."""
    print("Recording chunk...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmpfile.name, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return tmpfile.name

def transcribe_wav(wav_path: str) -> str:
    """Send wav to Groq Whisper and return transcript text."""
    with open(wav_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt="Specify context or spelling",
            response_format="verbose_json",
        )
    # Groq SDK returns an object with .text
    return getattr(transcription, "text", "")

def transcribe_audio_bytes(audio_bytes: bytes, sample_rate=16000) -> str:
    """
    Transcribe raw audio bytes using Groq Whisper.
    """
    import io
    
    # Create an in-memory bytes buffer
    audio_buffer = io.BytesIO()
    audio_buffer.name = "audio.wav" # Fake filename for the API
    
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    
    # Reset buffer position to start
    audio_buffer.seek(0)
    
    try:
        transcription = client.audio.transcriptions.create(
            file=audio_buffer,
            model="whisper-large-v3-turbo",
            prompt="Specify context or spelling and understand nepali language",
            response_format="verbose_json",
        )
        text = getattr(transcription, "text", "").strip()
        print("Transcript:", text)
        return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def recorder(stop_event: threading.Event, q: queue.Queue, duration=chunk_duration):
    """Continuously record chunks and enqueue temp wav paths."""
    while not stop_event.is_set():
        wav_path = record_chunk_to_wav(duration=duration)
        try:
            q.put(wav_path, timeout=0.1)
        except queue.Full:
            # Drop the oldest/this chunk if queue is full
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

def transcriber(stop_event, q: queue.Queue):
    while not stop_event.is_set() or not q.empty():
        try:
            wav_path = q.get(timeout=1)
        except queue.Empty:
            continue
        try:
            text = transcribe_wav(wav_path)
            if text:
                print("Transcript:", text)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

def main():
    """
    Continuous recording + background transcription.
    Press Ctrl+C to stop.
    """
    q = queue.Queue(maxsize=3)  # limit backlog
    stop_event = threading.Event()
    rec_thread = threading.Thread(target=recorder, args=(stop_event, q), daemon=True)
    tr_thread = threading.Thread(target=transcriber, args=(stop_event, q), daemon=True)

    print("Starting live transcription (continuous, no gaps)... Press Ctrl+C to stop.")
    rec_thread.start()
    tr_thread.start()

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    rec_thread.join()
    tr_thread.join()

if __name__ == "__main__":
    main()