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
from audio_recorder_streamlit import audio_recorder
from utils import audio_bytes_to_wav, speech_to_text, text_to_speech, get_llm_response, create_welcome_message
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


def main():
    st.set_page_config(page_title='Audio-based Chatbot')
    st.title("🎤 :blue[Voice Chatbot] 🤖💬")
    # st.sidebar.markdown("# Haven Wilder")
    st.sidebar.image('logo2.png', width=20, use_column_width=True)
    # st.sidebar.flag = 0

    # Initialize chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = []

    if "played_audios" not in st.session_state:
        st.session_state.played_audios = {}  # To track if an audio file has been played

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Handle the initial chat history setup
    if len(st.session_state.chat_history) == 0:
        welcome_audio_path = create_welcome_message()
        st.session_state.chat_history = [
            AIMessage(content="Hello, I can answer the questions about anything you are very curious about, and also about your private pdf files.", audio_file=welcome_audio_path)
        ]
        st.session_state.played_audios[welcome_audio_path] = False

    # Sidebar with mic button on top
    with st.sidebar:
        # Show "Speaking..." message during recording
        pdf_input_from_user = st.file_uploader("Upload the PDF file", type=['pdf'])
    

        if st.button("Create the Vector DB"):       
            # create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
            audio_bytes = audio_recorder(
                energy_threshold=0.01,
                pause_threshold=0.8,
                text="Speak now...max 5 min",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x"
                # Adjust the icon size
            )
            if pdf_input_from_user is not None:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(pdf_input_from_user.read())
                    pdf_file_path = temp_file.name
                # st.sidebar.flag = 1
                st.session_state.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})               
                st.session_state.loader = PyPDFLoader(pdf_file_path)
                st.session_state.text_document_from_pdf = st.session_state.loader.load()
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)              
                st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(st.session_state.text_document_from_pdf)
                st.session_state.vector_store = FAISS.from_documents(st.session_state.final_document_chunks, st.session_state.embeddings)
                st.success("Vector Store DB for this PDF file Is Ready")

                
            else:
                st.write("Please upload a PDF file first")
        

        else:
            audio_bytes = audio_recorder(
                energy_threshold=0.01,
                pause_threshold=0.8,
                text="Speak now...max 5 min",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x"
                # Adjust the icon size
            )
            st.info("")  # Clear the "Speaking..." message after recording

            if audio_bytes:
                # Save the user input audio file
                temp_audio_path = audio_bytes_to_wav(audio_bytes)
                if temp_audio_path:
                    # Transcribe the audio and update chat history
                    user_input = speech_to_text(audio_bytes)
                    st.session_state.chat_history.append(HumanMessage(content=user_input, audio_file=temp_audio_path))

                    # print("flag: " + str(st.sidebar.flag))
                    
                    #get retrieve data
                    if st.session_state.vector_store is not None:
                        retriever = st.session_state.vector_store.as_retriever()
                        retrieved_data = retriever.get_relevant_documents(user_input)
                        # print(user_input)
                        # print(retrieved_data)

                        # Generate AI response
                        response = get_llm_response(user_input, st.session_state.chat_history,retrieved_data)
                    
                    else:
                        no_retrieve_data = "there is no retrieved data, so answer the question from your information."
                        response = get_llm_response(user_input, st.session_state.chat_history,no_retrieve_data)
                    
                    # Convert the response to audio
                    audio_response = text_to_speech(response)

                    # Create an in-memory BytesIO stream
                    audio_stream = BytesIO()
                    audio_response.export(audio_stream, format="mp3")
                    audio_stream.seek(0)  # Rewind the stream to the beginning

                    # Save the AI response audio file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_response:
                        audio_stream.seek(0)
                        temp_audio_response.write(audio_stream.read())
                        audio_response_file_path = temp_audio_response.name

                    # Append AI response text and audio to history
                    st.session_state.chat_history.append(AIMessage(content=response, audio_file=audio_response_file_path))
                    st.session_state.played_audios[audio_response_file_path] = False  # Mark the new response as not played
                    audio_bytes = None

        # if st.button("New Chat"):
        #     # Save the current chat history to the chat_histories list
        #     st.session_state.chat_histories.append(st.session_state.chat_history)
        #     print(len(st.session_state.chat_history))
        #     # Initialize a new chat history with the default welcome message
        #     welcome_audio_path = create_welcome_message()
        #     st.session_state.played_audios[welcome_audio_path] = False
        #     # print("hey")
        #     st.session_state.chat_history = [
        #         AIMessage(content="Hello, I'm an AI assistant chatbot named Haven Wilder. How can I help you?", audio_file=welcome_audio_path)
        #     ]
            

        # if st.session_state.chat_histories:
        #     st.subheader("Chat History")
        #     for i, hist in enumerate(st.session_state.chat_histories):
        #         if st.button(f"Chat {i + 1}", key=f"chat_{i}"):
        #             st.session_state.chat_history = hist

    # Display the conversation history in the main area
    for message in st.session_state.chat_history:
        # print(message)
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                # Check if the audio file has already been played
                if hasattr(message, 'audio_file'):
                    if not st.session_state.played_audios.get(message.audio_file, False):
                        print("here1")
                        st.write(message.content)
                        st.audio(message.audio_file, format="audio/mp3", autoplay=True)
                        st.session_state.played_audios[message.audio_file] = True  # Mark as played
                    else:
                        print("here2")
                        st.write(message.content)
                        st.audio(message.audio_file, format="audio/mp3", autoplay=False)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                if hasattr(message, 'audio_file'):
                    st.write(message.content)
                    st.audio(message.audio_file, format="audio/wav", autoplay=False)

if __name__ == "__main__":
    main()
