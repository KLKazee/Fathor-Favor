import librosa
import soundfile as sf
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import os
import numpy as np
import io
import sys
import streamlit as st


# Downloads audio from youtube link and saves as mp3
def download_youtube_audio(url, output_name="output.mp3"):
    # Temporary filename
    temp_file = "temp_audio.webm"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": temp_file,
        "quiet": False
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Convert to MP3 using pydub
    audio = AudioSegment.from_file(temp_file)
    audio.export(output_name, format="mp3")

    # Clean up
    os.remove(temp_file)
    print(f"✅ Saved: {output_name}")
    return output_name

# Analyze and define the key of the audio file. Can Maybe add an option certain parts of the song
def define_key(mp3, mode):
    # Create audio prodile from downloaded mp3
    y, sr = librosa.load(mp3, sr=None)
    
    # Extract chroma features (12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Average across time to get a single vector representing the key
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean /= np.linalg.norm(chroma_mean)
    
    # Define major and minor key templates (Krumhansl-Schmuckler)
    major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize templates
    major_template /= np.linalg.norm(major_template)
    minor_template /= np.linalg.norm(minor_template)

    # Use known major/minor to be more accuratee. Too many polytones otherwise
    template = major_template if mode == "major" else minor_template
    
    # Compute correlation for each possible key (12 semitones)
    correlations = {}
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, note in enumerate(notes):
        corr = np.corrcoef(np.roll(template, i), chroma_mean)[0,1]
        correlations[f"{note} {mode}"] = corr
    
    # Return the key with highest correlation
    best_key = max(correlations, key=correlations.get)
    best_corr = correlations[best_key]
    
    return best_key, best_corr
   
# Calls saved mp3 file and shifts key by n_steps
def shift_key(input_file, n_steps, output_name):
    #Finding audio profile and sample rate
    y, sr = librosa.load(input_file)

    #Edited audio profile
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    os.remove(output_name)
    sf.write(output_name, y_shifted, sr)

    return

def findTones(orginalKey, newkey):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    orginalNote = orginalKey.split(" ")[0]
    orginalIndex = notes.index(orginalNote)
    newIndex = notes.index(newkey)
    steps = newIndex - orginalIndex
    return steps


# Display the selected option using success message
with st.form('key_trasition'):
    st.title("Change Key-inator")
    # Give title, probably easiler than using whatever title the uou tube video has
    st.subheader("Title of the song")
    title = st.text_input("Title:", placeholder = "Title")
    title = title + ".mp3"

    # Input youtube url
    st.subheader("Enter url from youtube. Use the url provided when using the share function")
    url = st.text_input("Youtube URL:", placeholder = "url")

    # Find mode to increase accuracy of key detection
    st.subheader("Select Mode of the original piece. This can be checked by playing a scale on an instrument along with the song")
    mode = st.radio("Select Mode:", ['Major', 'Minor'])

    # Input desired key
    st.subheader("Enter desired Key, program only exepts natural and sharp notes (e.g., C, D#, F)")
    newKey = st.text_input("Desired Key:", placeholder = "Key")

    st.write("*OPTIONAL: If the orginal key calculated is incorrect, or you already know the orginal key, manually enter it here.")
    originalKey = st.text_input("Orginal Key:", placeholder = "This is optional")
    
    submit = st.form_submit_button('Go')
    
if submit:
    if not url or not newKey:
        st.error("Please provide a valid YouTube URL and desired key.")
    else:
        if originalKey == "":
            with st.spinner("Finding and dowloading audio..."):
                download_youtube_audio(url, title)
            with st.spinner("Analyzing key..."):
                originalKey, corr = define_key(title, mode)
                st.header("Results")
                st.success(f"✅ Detected key: {originalKey} (correlation {corr:.3f})")
            with st.spinner("Adjusting the Key..."):
                steps = findTones(originalKey, newKey)
                shift_key(title, steps)
            
        else:
            with st.spinner("Finding and dowloading audio..."):
                download_youtube_audio(url, title)
            with st.spinner("Adjusting the Key..."):
                steps = findTones(originalKey, newKey)
                shift_key(title, steps)
                st.header("Results")


        st.info(f"Shifted by {steps:+} semitones to reach {newKey}.")

        col1, col2 = st.columns([3,1])
        with col1:
            st.audio(title + " - Edited.mp3", format="audio/mpeg")
        with col2:
            st.download_button("Download MP3", data=open(title, "rb"), file_name=title, mime="audio/mpeg")

    


