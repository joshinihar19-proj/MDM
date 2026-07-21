import cv2
import streamlit as st
import tempfile
import os

st.title("Motion Detector Web App")
st.write("Upload a video to detect motion using OpenCV.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties for saving the output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    # Temporary file for output video
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Controls in sidebar
    st.sidebar.header("Motion Settings")
    # Increase min_area default from 20 to 500 to ignore crowd/noise!
    min_area = st.sidebar.slider("Min Contour Area (Noise Filter)", 100, 5000, 800)

    stframe = st.empty()
    stop_button = st.button("Stop Processing")

    ret, prev_frame = cap.read()
    if ret:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur to smooth out pixel noise (crowd, flickering)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        while cap.isOpened() and not stop_button:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Convert and blur current frame
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
            
            # Calculate absolute difference
            diff = cv2.absdiff(prev_gray, current_gray)
            
            # Threshold & Dilate to fill in holes
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter out small noise based on slider
                if cv2.contourArea(contour) < min_area: 
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save frame to output video
            out.write(current_frame)

            # Display frame in Streamlit
            current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            stframe.image(current_frame_rgb, channels="RGB", use_column_width=True)
            
            prev_gray = current_gray

        cap.release()
        out.release()
        
        st.success("Processing complete!")

        # Allow user to download the generated motion-detected video
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Motion Video",
                data=file,
                file_name="motion_detected_output.mp4",
                mime="video/mp4"
            )
