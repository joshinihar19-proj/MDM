import cv2
import numpy as np
import streamlit as st
import tempfile
import os

st.title("Soccer Player & Ball Tracking App")
st.write("Detects players, classifies jersey colors (Red vs. Blue), and tracks the ball using HSV color segmentation.")

uploaded_file = st.file_uploader("Upload Match Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    
    # Extract video dimensions and frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    # Configure output video writer
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Sidebar Controls ---
    st.sidebar.header("Pitch & Detection Controls")
    
    # Grass HSV Bounds (Isolates pitch)
    st.sidebar.subheader("Grass Filter (HSV)")
    g_h_min, g_h_max = st.sidebar.slider("Grass Hue Range", 0, 180, (35, 75))
    
    # Team Jersey Thresholds
    st.sidebar.subheader("Jersey Pixel Thresholds")
    red_pixel_thresh = st.sidebar.slider("Red Team Sensitivity", 5, 50, 15)
    blue_pixel_thresh = st.sidebar.slider("Blue Team Sensitivity", 5, 50, 20)
    
    # Color Ranges in HSV
    green_low = np.array([g_h_min, 40, 40])
    green_high = np.array([g_h_max, 255, 255])

    blue_low = np.array([100, 50, 50])
    blue_high = np.array([140, 255, 255])

    # Red wraps around 0/180 in HSV
    red_low1 = np.array([0, 50, 50])
    red_high1 = np.array([10, 255, 255])
    red_low2 = np.array([170, 50, 50])
    red_high2 = np.array([180, 255, 255])

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])

    stframe = st.empty()
    stop_button = st.button("Stop Processing")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Convert frame to HSV and create Green Grass Mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv_frame, green_low, green_high)
        
        # 2. Extract non-grass objects on the pitch
        pitch_objects = cv2.bitwise_and(frame, frame, mask=grass_mask)
        gray_objects = cv2.cvtColor(pitch_objects, cv2.COLOR_BGR2GRAY)

        # 3. Threshold and apply morphological close to join broken contours
        _, thresh = cv2.threshold(gray_objects, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = np.ones((13, 13), np.uint8)
        thresh_morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 4. Find object contours
        contours, _ = cv2.findContours(thresh_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            # --- BALL DETECTION ---
            if (1 <= w <= 25) and (1 <= h <= 25):
                ball_crop = frame[y:y+h, x:x+w]
                if ball_crop.size == 0:
                    continue
                ball_hsv = cv2.cvtColor(ball_crop, cv2.COLOR_BGR2HSV)
                white_mask = cv2.inRange(ball_hsv, lower_white, upper_white)
                
                # Verify white pixel concentration for soccer ball
                if cv2.countNonZero(white_mask) >= 3:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Ball', (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- PLAYER DETECTION & JERSEY CLASSIFICATION ---
            elif (w > 15 and h >= 20) and (h >= 1.5 * w):
                player_crop = frame[y:y+h, x:x+w]
                if player_crop.size == 0:
                    continue
                player_hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)

                # Blue Jersey Mask
                blue_mask = cv2.inRange(player_hsv, blue_low, blue_high)
                
                # Red Jersey Mask (combining dual red ranges)
                red_mask1 = cv2.inRange(player_hsv, red_low1, red_high1)
                red_mask2 = cv2.inRange(player_hsv, red_low2, red_high2)
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)

                red_count = cv2.countNonZero(red_mask)
                blue_count = cv2.countNonZero(blue_mask)

                # Draw bounding box based on dominant team color
                if red_count >= red_pixel_thresh and red_count > blue_count:
                    # Red Team (Purple / Magenta Bounding Box)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (130, 0, 130), 2)
                    cv2.putText(frame, 'Red Team', (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 0, 130), 2)
                elif blue_count >= blue_pixel_thresh and blue_count > red_count:
                    # Blue Team (Blue Bounding Box)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
                    cv2.putText(frame, 'Blue Team', (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        # Write processed frame to output video file
        out.write(frame)

        # Render current frame in Streamlit (convert BGR to RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    out.release()

    st.success("Video processing finished!")

    # Provide download option for rendered video
    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Processed Soccer Video",
            data=file,
            file_name="soccer_detected_output.mp4",
            mime="video/mp4"
        )
