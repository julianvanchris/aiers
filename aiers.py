import torch
import torchaudio
import streamlit as st
import pyaudio
import wave
import soundfile as sf
import sounddevice as sd
import asyncio
import edge_tts
import numpy as np
from pydub import AudioSegment
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import unpad
import base64
import json
import nltk
import os

class AIERS:
    def __init__(self, config_file='stt_encrypted.json'):
        # Load NLTK resources
        nltk.download('punkt', quiet=True)

        # Decrypt and load configuration
        secret_key = st.secrets["secret_key"]
        decrypted_json = self._decrypt_json_file(config_file, secret_key)

        # Temporary file for Google Cloud credentials
        self._temp_key_path = 'temp_decrypted_key.json'
        with open(self._temp_key_path, 'w') as f:
            json.dump(decrypted_json, f)

        # Set Google Cloud credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._temp_key_path
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

        # Silero STT Model Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.decoder, self.utils = self._load_silero_model()
        
        # LLM Setup
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
        # Conversation Chain
        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=True,
            memory=ConversationBufferMemory()
        )

        # System prompt for the assistant
        self.system_prompt = """
        You are AIERS, the AI Emergency Response System. You assist both emergency responders (like 911 operators) and the callers during an emergency. Your responses must vary depending on the recipient of the message.

        When responding to emergency responders:
            - Provide concise and critical information.
            - Maintain professionalism and focus on actionable details.
            - Avoid emotional language, prioritize clarity, and describe the situation.

        When responding to the caller:
            - Offer reassurance and calm them down.
            - Provide guidance on what to do next and maintain a calming tone.
            - Encourage the caller to stay calm and follow instructions.
        """
        
        # TTS Configuration
        self.edge_tts_voice_dispatcher = "en-US-BrianNeural"
        self.edge_tts_voice_caller = "en-US-BrianNeural"
        self.edge_tts_output_file = "tts_output.wav"
        self.converted_wav_file = "converted_tts_output.wav"

    def _decrypt_json_file(self, encrypted_file, secret_key):
        """Decrypt JSON configuration file."""
        key = secret_key.encode('utf-8')[:16].ljust(16, b'\0')

        with open(encrypted_file, 'rb') as f:
            encrypted_data = base64.b64decode(f.read())

        iv = encrypted_data[:16]
        encrypted_data = encrypted_data[16:]

        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

        return json.loads(decrypted_data.decode('utf-8'))

    @staticmethod
    @st.cache_resource
    def _load_silero_model():
        """Load Silero model with caching"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, decoder, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language='en',
            device=device
        )
        return model, decoder, utils

    def cleanup_files(self):
        """Clean up temporary files."""
        try:
            files_to_delete = [self.edge_tts_output_file, self.converted_wav_file, self._temp_key_path]
            for file in files_to_delete:
                if os.path.exists(file):
                    os.remove(file)
        except Exception as e:
            st.error(f"Error cleaning up files: {e}")

    def transcribe_audio(self, audio_path, sample_rate=16000):
        try:
            # Use soundfile to read audio
            waveform, sr = sf.read(audio_path)
            
            # Convert to torch tensor
            waveform_tensor = torch.from_numpy(waveform).float()
            
            # Ensure mono channel
            if waveform_tensor.ndim > 1:
                waveform_tensor = waveform_tensor.mean(dim=1)
            
            # Resample if necessary
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform_tensor = resampler(waveform_tensor)
            
            # Ensure correct shape
            if waveform_tensor.dim() == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)
            
            # Prepare input for Silero model
            (read_batch, split_into_batches, read_audio, prepare_model_input) = self.utils
            input_tensor = prepare_model_input(waveform_tensor, device=self.device)
            
            # Transcribe
            with torch.no_grad():
                output = self.model(input_tensor)
                transcripts = [self.decoder(example.cpu()) for example in output]
            
            return ' '.join(transcripts)
        
        except Exception as e:
            st.error(f"Error in audio processing: {str(e)}")
            return "Transcription failed"

    async def _generate_tts(self, text, voice):
        """Generate speech using edge_tts and save it to a file."""
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(self.edge_tts_output_file)

    def convert_to_standard_wav(self):
        """Convert the edge_tts output to a standard WAV format using pydub."""
        try:
            audio = AudioSegment.from_file(self.edge_tts_output_file)
            audio.export(self.converted_wav_file, format="wav")
        except Exception as e:
            st.error(f"Error converting audio: {e}")

    def determine_role(self, user_input):
        """Identify whether the response is for 911 (dispatcher) or the caller."""
        # Simple logic based on the input; this can be expanded with more rules
        if "what should I do" in user_input.lower() or "help" in user_input.lower():
            return "caller"
        elif "emergency details" in user_input.lower():
            return "dispatcher"
        else:
            # Default response, could be expanded to handle more sophisticated rules
            return "caller"

    def text_to_speech(self, text, role="caller"):
        """Generate and play TTS audio for dispatcher or caller."""
        try:
            if role == "dispatcher":
                voice = self.edge_tts_voice_dispatcher
            else:
                voice = self.edge_tts_voice_caller
                
            # Generate TTS output asynchronously
            asyncio.run(self._generate_tts(text, voice))
            
            # Convert the generated audio to a standard WAV file
            self.convert_to_standard_wav()

            # Read the converted audio from the new WAV file
            wav_data, samplerate = sf.read(self.converted_wav_file)

            # Convert to float32 if necessary
            if wav_data.dtype != np.float32:
                wav_data = wav_data.astype(np.float32)

            # Normalize audio if needed
            if wav_data.max() > 1.0 or wav_data.min() < -1.0:
                wav_data = wav_data / max(abs(wav_data.max()), abs(wav_data.min()))
            
            # Play the audio using sounddevice
            sd.play(wav_data, samplerate=samplerate)
            sd.wait()
            
        except Exception as e:
            st.error(f"Text-to-Speech error: {e}")

        finally:
            # Clean up the output files after speaking
            self.cleanup_files()

    def get_llm_response(self, user_input, role="caller"):
        """Generate response based on role (dispatcher or caller)."""
        try:
            # Set system prompt based on role
            if role == "dispatcher":
                self.system_prompt += "\nRespond as if you're talking to emergency services."
            else:
                self.system_prompt += "\nRespond with reassurance and guidance for the caller."
            
            # Predict response using conversation chain
            ai_response = self.conversation.predict(input=user_input)
            return ai_response
        
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def __del__(self):
        self.cleanup_files()

    def record_audio(self, filename, duration=5):
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        # Open stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        st.write("🎙️ Recording...")
        frames = []
        # Record audio
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        st.write("✅ Recording stopped")
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()