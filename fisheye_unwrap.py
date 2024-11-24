import cv2
import numpy as np
import glob
import os

# Variables
video_or_folder_path = "/Users/elliott/Documents/SeaStereo/Software/Kyle_calibration/assets/footage/20240823_batray_F1_F2_S4_AC_630uT_EHVC.mp4"  # Path to video file or folder with multiple video files
calibration_file_path = "/Users/elliott/Documents/SeaStereo/Software/Kyle_calibration/calibrations/Z8+8-15mm_lens_11mm_1K.npz"  # Path to calibration file
output_path = "/Users/elliott/Documents/SeaStereo/Software/Kyle_calibration/output"  # Path to export unwrapped video

def calculate_scale_factor(input_dim, calibration_dim):
    scale_factor = min(input_dim[0] / calibration_dim[0], input_dim[1] / calibration_dim[1])
    return scale_factor

def undistort_fisheye_video(input_video, output_video, calibration_dim, K, D):
    calibration_dim = tuple(calibration_dim)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (frame_width, frame_height))

    scale_factor = calculate_scale_factor((frame_width, frame_height), calibration_dim)
    K_adj = K.copy()
    K_adj[0, 0] *= scale_factor
    K_adj[1, 1] *= scale_factor
    K_adj[0, 2] *= scale_factor
    K_adj[1, 2] *= scale_factor

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_adj, D, (frame_width, frame_height), np.eye(3))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_adj, D, np.eye(3), new_K, (frame_width, frame_height), cv2.CV_16SC2)

    print(f"Processing video: {input_video} ({total_frames} frames)")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out.write(undistorted_frame)
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.2f}% ({frame_count}/{total_frames} frames)", end="\r")

    print("\nVideo processing complete.")
    cap.release()
    out.release()

def process_videos(input_path, output_folder, K, D, calibration_dim):
    if os.path.isdir(input_path):
        video_files = glob.glob(f"{input_path}/*.mp4") + glob.glob(f"{input_path}/*.avi") + glob.glob(f"{input_path}/*.mov")
    else:
        video_files = [input_path]

    os.makedirs(output_folder, exist_ok=True)

    for video_file in video_files:
        output_video = os.path.join(output_folder, f"undistorted_{os.path.basename(video_file)}")
        undistort_fisheye_video(video_file, output_video, calibration_dim, K, D)


if __name__ == "__main__":
    # Load calibration data
    data = np.load(calibration_file_path)
    K = data["K"]
    D = data["D"]
    calibration_dim = tuple(data["calibration_dim"])
    
    process_videos(video_or_folder_path, output_path, K, D, calibration_dim)
