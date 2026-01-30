import pyaudio
import wave
import numpy as np

def record_audio_segment(filename: str = "live_input.wav", duration: float = 10.0, 
                         sample_rate: int = 48000, channels: int = 4, 
                         chunk_size: int = 1024) -> str:
    audio = pyaudio.PyAudio()
    stream = None
    try:
        device_index = None
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] >= channels:
                device_index = i
                break
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
        print(f"Recording {duration}s...", end="", flush=True)
        frames = []
        total_chunks = int(sample_rate / chunk_size * duration)
        for _ in range(total_chunks):
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
        print(" Done.")
        stream.stop_stream()
        stream.close()        
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.reshape(-1, channels).flatten()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2) 
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        return filename

    except Exception as e:
        print(f"\nRecording Error: {e}")
        return None
        
    finally:
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        audio.terminate()