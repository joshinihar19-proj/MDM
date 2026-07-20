import cv2
import streamlit as st
import tempfile

st.title("Motion Detector Web App")
st.write("Upload a video to detect motion using OpenCV.")

# File uploader for the video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # OpenCV requires a file path, so we save the uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    
    # Placeholder for the video stream
    stframe = st.empty()
    
    ret, prev_frame = cap.read()
    if ret:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_count = 0
        
        stop_button = st.button("Stop Processing")

        # Process frame by frame
        while cap.isOpened() and not stop_button:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and find difference
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, current_gray)
            
            # Apply threshold and find contours directly
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            for contour in contours:
                # 20 is very sensitive. You may want to increase this to 500+ for real-world videos.
                if cv2.contourArea(contour) < 20: 
                    continue
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Streamlit uses RGB, but OpenCV uses BGR. We must convert it before displaying.
            current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Update the placeholder with the new frame
            stframe.image(current_frame_rgb, channels="RGB", use_column_width=True)
            
            prev_gray = current_gray
            frame_count += 1
            
        cap.release()
        st.success("Video processing complete.")