import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import tempfile
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure the Gemini Pro API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro')

def analyze_squat(image):
    prompt = """
    Analyze this image of a person performing a squat exercise. Provide the following details:
    1. Overall form rating on a scale of 1-10, where 10 is perfect form.
    2. Breakdown of the form:
       - Back position (1-10)
       - Knee alignment (1-10)
       - Depth of squat (1-10)
       - Balance and stability (1-10)
    3. Brief comment on the form in this specific frame.
    
    Format the response as JSON with keys 'overall_rating', 'breakdown', and 'comment'.
    """
    try:
        response = model.generate_content([prompt, image])
        
        if not response.text:
            raise ValueError("Empty response from API")
        
        try:
            analysis = json.loads(response.text)
        except json.JSONDecodeError:
            analysis = {
                'overall_rating': 5,
                'breakdown': {
                    'Back position': 5,
                    'Knee alignment': 5,
                    'Depth of squat': 5,
                    'Balance and stability': 5
                },
                'comment': response.text
            }
        
        if not all(key in analysis for key in ['overall_rating', 'breakdown', 'comment']):
            raise ValueError("Response missing required keys")
        
        return analysis
    except Exception as e:
        st.error(f"Error in analyze_squat: {str(e)}")
        st.error(f"API Response: {response.text if 'response' in locals() else 'No response'}")
        
        return {
            'overall_rating': 5,
            'breakdown': {
                'Back position': 5,
                'Knee alignment': 5,
                'Depth of squat': 5,
                'Balance and stability': 5
            },
            'comment': f"Error occurred during analysis: {str(e)}"
        }

def create_radar_chart(breakdown):
    categories = list(breakdown.keys())
    values = list(breakdown.values())
    
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 10)
    plt.title("Average Squat Form Breakdown")
    return fig

def process_video(video_file, max_frames, analysis_interval):
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(video_file.read())
        temp_filename = tfile.name

    try:
        cap = cv2.VideoCapture(temp_filename)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        analyze_interval = fps * analysis_interval  # Convert seconds to frames
        
        overall_ratings = []
        breakdowns = defaultdict(list)
        comments = []
        
        progress_bar = st.progress(0)
        
        for i in range(0, min(frame_count, max_frames * analyze_interval), analyze_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try:
                    analysis = analyze_squat(pil_image)
                    overall_ratings.append(analysis['overall_rating'])
                    for key, value in analysis['breakdown'].items():
                        breakdowns[key].append(value)
                    comments.append(analysis['comment'])
                except Exception as e:
                    st.error(f"An error occurred during analysis of frame {i}: {str(e)}")
            
            progress_bar.progress((i + 1) / min(frame_count, max_frames * analyze_interval))
        
        cap.release()
    finally:
        try:
            os.unlink(temp_filename)
        except Exception as e:
            st.warning(f"Could not delete temporary file: {str(e)}")

    return overall_ratings, breakdowns, comments, analyze_interval

def display_results(overall_ratings, breakdowns, comments, analyze_interval):
    avg_overall_rating = sum(overall_ratings) / len(overall_ratings)
    avg_breakdowns = {k: sum(v) / len(v) for k, v in breakdowns.items()}
    
    st.subheader("Video Analysis Results")
    st.write(f"Average Overall Form Rating: {avg_overall_rating:.2f}/10")
    
    st.subheader("Average Form Breakdown")
    for key, value in avg_breakdowns.items():
        st.write(f"- {key}: {value:.2f}/10")
    
    st.subheader("Average Form Breakdown Visualization")
    radar_chart = create_radar_chart(avg_breakdowns)
    st.pyplot(radar_chart)
    
    st.subheader("Frame-by-Frame Comments")
    for i, comment in enumerate(comments):
        st.write(f"Frame {i * analyze_interval}: {comment}")
    
    st.subheader("Overall Rating Over Time")
    fig, ax = plt.subplots()
    ax.plot(range(0, len(overall_ratings) * analyze_interval, analyze_interval), overall_ratings)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Overall Rating")
    ax.set_title("Squat Form Rating Throughout the Video")
    st.pyplot(fig)

def main():
    st.title("Squat Form Analyzer (Limited API Usage)")

    st.write("Note: This version limits API usage to conserve your free quota.")

    analysis_mode = st.radio("Choose analysis mode:", ("Upload Video", "Use Webcam"))

    max_frames = st.slider("Maximum number of frames to analyze:", 1, 10, 5)
    analysis_interval = st.slider("Analyze every N seconds:", 1, 10, 5)

    if analysis_mode == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            if st.button('Analyze Video'):
                with st.spinner(f'Analyzing up to {max_frames} frames...'):
                    overall_ratings, breakdowns, comments, analyze_interval = process_video(uploaded_file, max_frames, analysis_interval)
                display_results(overall_ratings, breakdowns, comments, analyze_interval)
    
    elif analysis_mode == "Use Webcam":
        st.write("Webcam Feed:")
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()
        
        capture_button = st.button('Capture and Analyze Frame')
        stop_button = st.button('Stop Webcam')
        
        while not stop_button:
            ret, frame = video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame)
                
                if capture_button:
                    pil_image = Image.fromarray(frame)
                    with st.spinner('Analyzing frame...'):
                        try:
                            analysis = analyze_squat(pil_image)
                            st.write(f"Overall Form Rating: {analysis['overall_rating']}/10")
                            st.write("Form Breakdown:")
                            for key, value in analysis['breakdown'].items():
                                st.write(f"- {key}: {value}/10")
                            st.write(f"Comment: {analysis['comment']}")
                        except Exception as e:
                            st.error(f"An error occurred during analysis: {str(e)}")
                    break
        
        video_capture.release()

if __name__ == "__main__":
    main()

st.write("Note: Make sure you have set the GOOGLE_API_KEY environment variable.")