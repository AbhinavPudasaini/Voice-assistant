import collections
import queue
import sys
import threading
import time
import wave
import numpy as np
import sounddevice as sd
import webrtcvad

class VADRecorder:
    def __init__(self, sample_rate=16000, frame_duration_ms=20, padding_duration_ms=700, vad_level=3):
        """
        :param sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000 for webrtcvad)
        :param frame_duration_ms: Frame duration in ms (must be 10, 20, or 30)
        :param padding_duration_ms: Amount of silence to wait before deciding speech has ended
        :param vad_level: Aggressiveness of VAD (0-3). 3 is most aggressive (filters out more noise).
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.vad = webrtcvad.Vad(vad_level)
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.stop_event = threading.Event()
        
    def _read_audio_stream(self):
        """Generator that yields audio frames from the microphone."""
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', blocksize=self.frame_size) as stream:
            while not self.stop_event.is_set():
                data, overflowed = stream.read(self.frame_size)
                if overflowed:
                    print("Audio buffer overflow", file=sys.stderr)
                # Convert numpy array to raw bytes for webrtcvad
                yield data.tobytes()

    def listen(self):
        """
        Listens for a single segment of speech.
        Returns:
            bytes: The complete audio data of the speech segment (PCM 16-bit mono).
        """
        num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        
        print("Listening... (Speak now)")
        
        stream = self._read_audio_stream()
        
        for frame in stream:
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If more than 90% of the frames in the ring buffer are speech, trigger start
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    print("Speech detected! Recording...")
                    # Include the padding (start of speech)
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are silence, trigger end
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    print("Silence detected. Processing...")
                    triggered = False
                    # Yield the accumulated audio
                    return b''.join(voiced_frames)
        
        return b''

    def close(self):
        self.stop_event.set()