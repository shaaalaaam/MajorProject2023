import cv2
import time

# Create a VideoCapture object to capture video from an mp4 file
cap = cv2.VideoCapture(0)

# Check if the mp4 file was opened successfully
if not cap.isOpened():
    print("Could not open video file")
    exit()

# Define the number of seconds between each frame capture
capture_interval = 10

# Get the current time
start_time = time.time()

# Loop to read frames from the mp4 file and display them
while True:
    # Read a frame from the mp4 file
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("End of video file")
        break

    # Get the current time
    current_time = time.time()

    # Check if the elapsed time is equal to or greater than the capture interval
    if current_time - start_time >= capture_interval:
        # Capture the frame and save it as an image file
        cv2.imwrite('captured_frame.jpg', frame)
        print("Frame captured")

        # Update the start time
        start_time = current_time

    # Display the frame in the same window
    cv2.imshow('Video', frame)

    # Set the frame rate to 10 frames per second
    '''if cv2.waitKey(100) == ord('q'):
        break'''

# Release the VideoCapture object
cap.release()

# Close the window
cv2.destroyAllWindows()
