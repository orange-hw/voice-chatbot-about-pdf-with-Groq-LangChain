
# utils.py
import time
import requests
import tempfile
import re
from io import BytesIO
from gtts import gTTS
from pydub import AudioSegment
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os 
# Load the API key from Streamlit secrets
# api_key = st.secrets['secrets']["API_KEY"]
api_key = "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"

# Initialize the Groq client
client = Groq(api_key=api_key)

# Initialize the Groq model for LLM responses
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=api_key, max_tokens=500)  # Limit tokens to 100

def audio_bytes_to_wav(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            # Downsample to reduce file size if needed
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_wav.name, format="wav")
            return temp_wav.name
    except Exception as e:
        st.error(f"Error during WAV file conversion: {e}")
        return None

def split_audio(file_path, chunk_length_ms):
    audio = AudioSegment.from_wav(file_path)
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

def speech_to_text(audio_bytes):
    try:
        # Convert the audio bytes to WAV
        temp_wav_path = audio_bytes_to_wav(audio_bytes)

        if temp_wav_path is None:
            return "Error"

        # Increase file size limit
        if os.path.getsize(temp_wav_path) > 50 * 1024 * 1024:
            st.error("File size exceeds the 50 MB limit. Please upload a smaller file.")
            return "Error"

        # Define chunk length (e.g., 5 minutes = 5 * 60 * 1000 milliseconds)
        chunk_length_ms = 5 * 60 * 1000
        chunks = split_audio(temp_wav_path, chunk_length_ms)

        transcription = ""
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk:
                chunk.export(temp_chunk.name, format="wav")
                with open(temp_chunk.name, "rb") as file:
                    chunk_transcription = client.audio.transcriptions.create(
                        file=("audio.wav", file.read()),
                        model="whisper-large-v3",
                        response_format="text",
                        language="en",
                        temperature=0.0,

                    )
                    transcription += chunk_transcription + " "

        return transcription.strip()
    except Exception as e:
        st.error(f"Error during speech-to-text conversion: {e}")
        return "Error"


def text_to_speech(text, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            # Enforce English language for text-to-speech
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                tts.save(f.name)
                audio = AudioSegment.from_mp3(f.name)
            return audio
        except requests.ConnectionError as e:
            attempt += 1
            if attempt < retries:
                st.warning(f"Internet connection issue. Retrying ({attempt}/{retries})...")
                time.sleep(delay)  # Wait before retrying
            else:
                st.error(f"Failed to connect after {retries} attempts. Please check your internet connection.")
                # Return silent audio in case of error
                return AudioSegment.silent(duration=1000)
        except Exception as e:
            st.error(f"Error during text-to-speech conversion Please check your internet connection: {e}")
            # Return silent audio in case of other errors
            return AudioSegment.silent(duration=1000)

def remove_punctuation(text):
    # Remove punctuation from the text
    return re.sub(r'[^\w\s]', '', text)

def get_llm_response(query, chat_history, retrieve_data):
    try:
        # Updated template with detailed guidelines
        # template = """
        #        You are a highly qualified female psychiatrist assistant chatbot named "psychologist
        #        " with extensive experience in mental health. Your role is to provide professional, empathetic, and culturally authentic advice and answers to the user's questions. You communicate using everyday language, incorporating local idioms and expressions while avoiding loanwords from other languages.

        #         Your expertise is exclusively in providing information and advice related to mental health. If a question falls outside the scope of mental health, you should respond with, "I specialize only in mental health-related queries."

        #         Key Considerations:

        #         Professionalism: Maintain a high level of expertise and ethical standards.
        #         Cultural Authenticity: Understand and reflect the values, beliefs, and customs relevant to the user’s cultural context.
        #         Empathy: Show genuine compassion and support for the user's emotional well-being.
        #         You do not provide information outside of this 
        #           scope. If a question is not about psychologist,mental health, respond with,
        #          "I specialize only in psychologist,mental health related queries."                 
        #          **Chat History:** {chat_history}

        #         **User:** {user_query}

        # """


        # template = """
        #        You are a highly qualified assistant chatbot named "haven wilder
        #        " with extensive experience in AI industry. Your role is to provide professional advice and answers to the user's questions. You communicate using everyday language, incorporating local idioms and expressions while avoiding loanwords from other languages.

        #         Your expertise is exclusively in providing information and advice related to AI field.
        #         answer the client's question in 50 words. 
                                  
        #          **Chat History:** {chat_history}

        #         **User:** {user_query}

        # """

        template = """
                You are a helpful assistant. Below is some information:

                **retrieve_data**: {retrieve_data}

                **Chat History**: {chat_history}

                **User:** {user_query}

                When a user asks a question:

                If the question is related to the PDF, retrieve the answer directly from retrieve_data.
                If the question is unrelated to the PDF, answer using your general knowledge from the LLM training.
                Use chat_history to maintain the conversational context and continuity.
                If you cannot find a precise answer, respond politely and ask for clarification if needed.
                Your role is to provide professional advice and answers to the user's questions. You communicate using everyday language, incorporating local idioms and expressions while avoiding loanwords from other languages.
                Answer the client's question in 100 words.
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        response_gen = chain.stream({
            "chat_history": chat_history,
            "user_query": query,
            "retrieve_data":retrieve_data
        })

        # Combine all parts of the response and apply text clean-up
        response_text = ''.join(list(response_gen))
        # response_text = remove_punctuation(response_text)

        # Remove repeated text
        response_lines = response_text.split('\n')
        unique_lines = list(dict.fromkeys(response_lines))  # Remove duplicates while preserving order
        cleaned_response = '\n'.join(unique_lines)

        return cleaned_response
    except Exception as e:
        st.error(f"Error during LLM response generation: {e}")
        return "Error"

def create_welcome_message():
    welcome_text = "Hello, I can answer the questions about anything you are very curious about, and also about your private pdf files."  #  greeting with female pronoun
    tts = gTTS(text=welcome_text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        tts.save(f.name)
        return f.name



