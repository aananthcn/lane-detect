#!/usr/bin/env python3
"""
Lane Detection using UFLD-v1 - FINAL STABLE VERSION
- Corrects MPS float32 conversion order
- Implements explicit ImageNet normalization
- Fixes [1, 101, 56, 4] output parsing to prevent zig-zags
"""

import cv2
import numpy as np
import sys
import socket
import os
import argparse
import torch
from scipy.special import softmax

# TuSimple row anchors (56 values from 64 to 284 step 4)
tusimple_row_anchor = list(range(64, 288, 4))

# === SETUP CHECKS ===
UFDLD_PATH = './ufld'
MODEL_PATH = f'{UFDLD_PATH}/model/weights/tusimple_res18.pth'

print("Checking UFLD setup...")
if not os.path.exists(UFDLD_PATH) or not os.path.exists(MODEL_PATH):
    print("ERROR: Folder or model missing.")
    sys.exit(1)

sys.path.insert(0, UFDLD_PATH)
try:
    from model.model import parsingNet
    print("✓ UFLD modules imported successfully!")
except ImportError as e:
    print(f"ERROR importing UFLD: {e}")
    sys.exit(1)

# === UTILITY FUNCTIONS ===
def check_udp_port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", int(port)))
        sock.close()
        return False
    except OSError:
        return True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_type', choices=['udp', 'file'])
    parser.add_argument('source')
    return parser.parse_args()

def get_video_source(input_type, source):
    if input_type == 'udp':
        if check_udp_port_in_use(source):
            print(f"ERROR: Port {source} in use.")
            sys.exit(1)
        pipeline = f"udpsrc port={source} ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        if not os.path.isfile(source):
            print(f"ERROR: File '{source}' not found.")
            sys.exit(1)
        return cv2.VideoCapture(source)

# === MODEL LOADING ===
def load_ufld_model():
    net = parsingNet(pretrained=False, backbone='18', cls_dim=(101, 56, 4), use_aux=False)
    
    # Load to CPU first to ensure compatibility
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = ckpt if 'state_dict' not in ckpt else ckpt['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    
    # Select best available device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    print(f"✓ Model loaded on {device}")
    return net, device

# === INFERENCE ENGINE ===
def ufld_detect(net, device, frame):
    h, w = frame.shape[:2]
    
    # 1. Precise Pre-processing
    img = cv2.resize(frame, (800, 288))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 explicitly before math to avoid double precision issues
    img = img.astype(np.float32) / 255.0
    
    # ImageNet Normalization - Crucial for accurate detection
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Reorder to (Batch, Channel, Height, Width)
    img = img.transpose(2, 0, 1)[None, ...]
    
    # IMPORTANT: Cast to float() ON CPU before moving to device
    img_tensor = torch.from_numpy(img).float().to(device)

    with torch.no_grad():
        out = net(img_tensor)

    # 2. Output Decoding
    # Output shape: [1, 101, 56, 4] -> Squeeze to [101, 56, 4]
    loc = out.detach().cpu().numpy()[0]
    
    # Transpose to [Lanes, Grids, Rows] -> [4, 101, 56]
    loc = loc.transpose(2, 0, 1)

    # Apply Softmax along the grid dimension (axis 1)
    prob = softmax(loc, axis=1)
    
    # Column mapping for the 100 horizontal grid cells
    col_sample = np.linspace(0, 800 - 1, 100)

    lanes = []
    for i in range(4):
        # Ignore index 100 (background/no-lane class)
        lane_probs = prob[i, :100, :] 
        
        # Find highest probability grid for each of the 56 rows
        grid_idx = np.argmax(lane_probs, axis=0) 
        conf = np.max(lane_probs, axis=0)         

        points = []
        for j in range(56):
            # Threshold of 0.4 is safe with proper normalization
            if conf[j] > 0.014:
                x = int(col_sample[grid_idx[j]] * w / 800)
                y = int(tusimple_row_anchor[j] / 288.0 * h)
                points.append([x, y])

            else:
                print("Confidence: " + str(conf[j]))
        
        if len(points) > 5:
            lanes.append(np.array(points))

    return lanes

def display_lanes(frame, lanes):
    overlay = frame.copy()
    for lane in lanes:
        if len(lane) >= 2:
            # Draw thick green lanes
            cv2.polylines(overlay, [lane.astype(int)], False, (0, 255, 0), thickness=8)
    # Blend the lane markings with the original frame
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

# === MAIN RUNNER ===
def main():
    args = parse_arguments()
    cap = get_video_source(args.input_type.lower(), args.source)
    if not cap.isOpened():
        print("Error opening video source")
        sys.exit(1)

    net, device = load_ufld_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lanes = ufld_detect(net, device, frame)
        result = display_lanes(frame, lanes)
        
        cv2.imshow("UFLD Lane Detection - Press q to quit", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Process Complete.")

if __name__ == "__main__":
    main()