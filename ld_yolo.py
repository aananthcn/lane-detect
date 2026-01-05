import cv2
import numpy as np
import sys
import socket
import os
import argparse
from ultralytics import YOLO

# Always use the first display
os.environ["DISPLAY"] = ":0"

# Function to check if UDP port is available
def check_udp_port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", int(port)))
    except OSError:
        return True
    finally:
        sock.close()
    return False

# Argument Parsing (Keeping your original UI)
def parse_arguments():
    parser = argparse.ArgumentParser(description='Lane Detection using YOLO11 with UDP or File input')
    parser.add_argument('input_type', type=str, choices=['udp', 'file'], help='Type of input: "udp" or "file"')
    parser.add_argument('source', help='Port number for UDP or video file path for file input')
    
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

# Get video source (Keeping your original UI)
def get_video_source(input_type, source):
    if input_type == 'udp':
        if check_udp_port_in_use(source):
            print(f"ERROR: Port {source} in use. Please stop the other application and retry!")
            sys.exit(1)
        return cv2.VideoCapture(f"udp://0.0.0.0:{source}")
    else:
        return cv2.VideoCapture(source)

def main():
    args = parse_arguments()
    input_type = args.input_type.lower()

    # Load the latest YOLO11 segmentation model
    # Note: For specific lane detection, you would typically use a model fine-tuned on lanes.
    # 'yolo11n-seg.pt' is the standard pretrained model for general segmentation.
    model = YOLO("yolo11n-seg.pt")

    cap = get_video_source(input_type, args.source)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.source}'")
        sys.exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO11 inference
        # We use stream=True for better memory management in video processing
        results = model.predict(frame, conf=0.25, show=False, stream=True)

        for result in results:
            # Get the annotated frame (includes segmentation masks and boxes)
            annotated_frame = result.plot()
            
            # Display the result
            cv2.imshow("YOLO11 Lane Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()