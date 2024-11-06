# Import libraries
import os
import whisper
from groq import Groq
import streamlit as st
from yt_dlp import YoutubeDL

# Access the secret API key
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Define function to download audio from YouTube using yt-dlp
def download_audio_from_youtube(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return "temp_audio.mp3"

# Define function to transcribe and summarize video
def summarize_video(video_path):
    # Load Whisper model
    whisper_model = whisper.load_model("base")
    
    # Transcribe video audio
    transcription = whisper_model.transcribe(video_path)
    text_content = transcription['text']
    
    # Use Groq API to summarize the transcription
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Summarize this: {text_content}"}
        ],
        model="llama3-8b-8192",
    )
    
    # Extract summarized content
    summary = chat_completion.choices[0].message.content
    return summary

# Streamlit app for video summarization
def main():
    st.title("Video Summarizer")

    # Option to upload a file or use a YouTube link
    option = st.radio("Choose an option:", ("Upload Video", "YouTube Link"))

    if option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            # Save uploaded video file temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.read())
            
            # Transcribe and summarize the video
            st.write("Transcribing and summarizing...")
            summary = summarize_video("temp_video.mp4")
            st.write("**Summary:**")
            st.write(summary)

    elif option == "YouTube Link":
        youtube_url = st.text_input("Enter YouTube Video URL")
        if youtube_url:
            # Download audio from YouTube
            st.write("Downloading audio...")
            video_path = download_audio_from_youtube(youtube_url)
            
            # Transcribe and summarize the downloaded audio
            st.write("Transcribing and summarizing...")
            summary = summarize_video(video_path)
            st.write("**Summary:**")
            st.write(summary)

# Run Streamlit app
if __name__ == "__main__":
    main()
