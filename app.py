import cv2
import streamlit as st
import tempfile
import os

st.title("Motion Detector Web App")
st.write("Upload a video to detect motion. Optimized for sports/panning cameras.")

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
    min_area = st.sidebar.slider("Min Contour Area (Ignores noise)", 100, 2000, 300)
    max_area = st.sidebar.slider("Max Contour Area (Ignores goalposts/lines)", 1000, 10000, 3000)
    mask_percentage = st.sidebar.slider("Top Screen Masking % (Ignores audience)", 0, 50, 30)

    stframe = st.empty()
    stop_button = st.button("Stop Processing")

    # Use MOG2 Background Subtraction (Handles panning and minor shifts better than absdiff)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    while cap.isOpened() and not stop_button:
        ret, current_frame = cap.read()
        if not ret:
            break
        
        # Calculate the top portion of the screen to mask out (where the crowd is)
        roi_top = int(height * (mask_percentage / 100.0))

        # Apply MOG2 subtractor
        fg_mask = backSub.apply(current_frame)
        
        # Black out the top section of the mask so we ignore audience movement completely
        fg_mask[0:roi_top, 0:width] = 0
        
        # Clean up the mask (remove noise, fill holes)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter out tiny noise AND massive background shifts
            if area < min_area or area > max_area: 
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save frame to output video
        out.write(current_frame)

        # Display frame in Streamlit
        current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        stframe.image(current_frame_rgb, channels="RGB", use_column_width=True)

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
