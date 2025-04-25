# CELL 1: Imports and Loading Indicator
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets

def loading_indicator(duration=0.1):
    """Simple loading indicator that shows a spinning slash"""
    chars = ['/', '-', '\\', '|']
    for i in range(4):
        sys.stdout.write('\r' + 'Loading ' + chars[i % len(chars)])
        sys.stdout.flush()
        time.sleep(duration)

# CELL 2: Test Print Statements
print("Running!")
print("Please ensure the video is trimmed to the most stable part before uploading.")
print("This will help improve the accuracy of the lava flow analysis.")

# CELL 3: Process Video Function
def process_video(video_path, frame_interval=5):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = frame_interval / fps  # Time interval between frames
    
    # Initialize variables to store lava flow properties
    frame_count = 0
    flow_speeds = []
    flow_widths = []
    
    # Progress bar
    progress = widgets.FloatProgress(value=0, min=0, max=total_frames, description='Processing:')
    display(progress)
    
    # Start time for estimating remaining time
    start_time = time.time()
    
    # Progress checkpoints (25%, 50%, 75%, 100%)
    checkpoints = [0.25, 0.5, 0.75, 1.0]
    checkpoint_frames = [int(total_frames * cp) for cp in checkpoints]
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        loading_indicator()
        
        # Skip frames based on the interval
        for _ in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            break
        
        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow (velocity vectors between frames)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate the magnitude of the flow vectors (speed)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_speed = np.mean(magnitude)  # Average speed across the frame
        
        # Calculate flow width (placeholder logic)
        flow_width = gray.shape[1]  # Placeholder for flow width
        
        flow_speeds.append(mean_speed)
        flow_widths.append(flow_width)
        
        # Update progress bar
        frame_count += frame_interval
        progress.value = frame_count
        
        # Progress at checkpoints
        if frame_count in checkpoint_frames:
            elapsed_time = time.time() - start_time
            avg_time_per_frame = elapsed_time / frame_count
            remaining_frames = total_frames - frame_count
            remaining_time = remaining_frames * avg_time_per_frame
            
            # Clear previous output
            clear_output(wait=True)
            display(progress)
            
            # Progress message
            progress_percent = int((frame_count / total_frames) * 100)
            print(f"Progress: {progress_percent}% complete")
            print(f"Processed {frame_count}/{total_frames} frames")
            print(f"Estimated time remaining: {remaining_time:.2f} seconds")
            print("------")
        
        # Update previous frame
        prev_gray = gray
    
    # Release the video capture object
    cap.release()
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(flow_speeds, label='Flow Speed')
    plt.title('Lava Flow Speed Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Speed (Pixels per Second)')
    
    plt.subplot(1, 2, 2)
    plt.plot(flow_widths, label='Flow Width')
    plt.title('Lava Flow Width Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Width (Pixels)')
    
    plt.tight_layout()
    plt.show()

# CELL 4: Display Frame and Contour Function
def display_frame_and_contour(video_path, frame_number=0):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Set the video to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a blank image
    contour_map = np.zeros_like(frame)
    cv2.drawContours(contour_map, contours, -1, (0, 255, 0), 2)
    
    # Release the video capture object
    cap.release()
    
    # Display the grayscale frame and contour map
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Frame')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(contour_map)
    plt.title('Contour Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# CELL 5: Calculate Velocity Map Function
def calculate_velocity_map(video_path, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    velocity_maps = []
    
    while True:
        loading_indicator()
        
        # Skip frames based on the interval
        for _ in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Compute the magnitude of the flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        velocity_maps.append(magnitude)
        
        prev_gray = gray
    
    cap.release()
    
    return velocity_maps

# CELL 6: Visualize Lava Flow Function
def visualize_lava_flow(video_path, frame_number=0, threshold=128):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Set the video to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to identify lava regions
    _, lava_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Create a color visualization (red = lava, black = not lava)
    lava_visualization = np.zeros_like(frame)
    lava_visualization[lava_mask == 255] = [0, 0, 255]  # Red for lava
    
    # Display the result
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Frame')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(lava_visualization)
    plt.title('Lava Visualization (Blue = Lava)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# CELL 7: Example Usage
video_file_path = "lava flow 2.mp4"  # Replace with the actual file name

# Display a frame and its contour map
display_frame_and_contour(video_file_path, frame_number=0)

# Calculate and display velocity map
velocity_maps = calculate_velocity_map(video_file_path, frame_interval=5)
if velocity_maps:
    plt.imshow(velocity_maps[0], cmap='hot')
    plt.title("Velocity Map")
    plt.colorbar(label="Velocity Magnitude")
    plt.axis('off')
    plt.show()

# Visualize lava flow
visualize_lava_flow(video_file_path, frame_number=0, threshold=128)

# Process the entire video
process_video(video_file_path, frame_interval=5) 