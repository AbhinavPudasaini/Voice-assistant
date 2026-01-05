from controller import call_agent, decider
from speech_to_speech2 import transcribe_audio_bytes
from text_to_speech import stream_text_to_speech, play_audio_file
from vad_recorder import VADRecorder
import threading
import sys
import uuid
import os

#  NOTE : : A DECIDER LLM SHOULD BE USE TO DECIDE WHEN TO START ANSWERING.

def _quit_listener(stop_event: threading.Event):
    import msvcrt
    while not stop_event.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch.lower() == "q":
                stop_event.set()
                return

def run():
    stop_event = threading.Event()
    threading.Thread(target=_quit_listener, args=(stop_event,), daemon=True).start()
    
    # Initialize VAD Recorder
    recorder = VADRecorder()
    
    # Buffer for accumulated speech
    full_text_buffer = []
    
    # Unique session ID for this conversation
    session_id = str(uuid.uuid4())
    interrupt_event = threading.Event()
    
    # Play initial greeting
    initial_audio = "Practices/VoiceAgent/initial.mp3"
    if os.path.exists(initial_audio):
        print("Playing initial greeting...")
        play_audio_file(initial_audio)
    else:
        print(f"Warning: {initial_audio} not found. Skipping initial greeting.")
    
    print(f"Running voice loop (Session: {session_id}). Press 'q' to quit.")
    
    try:
        while not stop_event.is_set():
            # 1. Listen (Blocks until speech is detected)
            audio_data = recorder.listen()
            
            # Signal background TTS to stop immediately when user starts talking
            interrupt_event.set()
            
            if stop_event.is_set():
                break
                
            if not audio_data:
                continue
                
            # 2. Transcribe
            text = transcribe_audio_bytes(audio_data)
            if not text or len(text.strip()) < 2:
                continue
            
            full_text_buffer.append(text)
            combined_text = " ".join(full_text_buffer)
            print(f"User said (partial): {text}")
            
            # 3. Decider (Intelligence + Context)
            intent = decider(combined_text, session_id=session_id)
            print(f"Intent: {intent}")
            
            if intent == "READY_FOR_RESPONSE":
                # 4. Generate Response
                agent_output = call_agent(combined_text, session_id=session_id)
                print(f"Agent response: {agent_output}")
                
                # 5. Clear interrupt flag and start TTS in background thread
                interrupt_event.clear()
                threading.Thread(
                    target=stream_text_to_speech,
                    args=(agent_output,),
                    kwargs={"interrupt_event": interrupt_event},
                    daemon=True
                ).start()
                
                # Clear buffer after successful response
                full_text_buffer.clear()
                
            elif intent == "USER_INTERRUPTING":
                print("User interrupted. Clearing buffer.")
                full_text_buffer.clear()
                
            elif intent == "USER_NOT_DONE":
                print("User not done. Waiting for more...")
                continue
            
            elif intent == "END_CaLL":
                print("User requested to end the call.")
                break
            else:
                continue
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error in voice loop: {e}")
    finally:
        # Play final goodbye message
        final_audio = "Practices/VoiceAgent/last.mp3"
        if os.path.exists(final_audio):
            print("Playing goodbye message...")
            play_audio_file(final_audio)
        else:
            print(f"Warning: {final_audio} not found. Skipping goodbye message.")
        
        recorder.close()
        print("Stopped.")

if __name__ == "__main__":
    run()