import cv2
import numpy as np

# Variables
video_file_path = "/Users/elliott/Documents/SeaStereo/Software/Kyle_calibration/assets/footage/fisheye_test_Z8+8-15mm_lens_11mm_1K_120fps_1920x1080.mp4"  # Path to input video
checkerboard_size = (13, 8)  # Length and width of the checkerboard pattern
calibration_output_path = "/Users/elliott/Documents/SeaStereo/Software/Kyle_calibration/calibrations/Z8+8-15mm_lens_11mm_1K.npz"  # Output path for calibration file
frame_interval = 60  # Number of frames to skip between checks

def calibrate_fisheye_video(video_file, checkerboard, frame_interval=30):
    """
    Calibrate a fisheye camera using a video file containing a checkerboard pattern.

    Parameters:
        video_file (str): Path to the input video file.
        checkerboard (tuple): Dimensions of the checkerboard pattern (rows, columns).
        frame_interval (int): Number of frames to skip for calibration.

    Returns:
        tuple: Camera matrix, distortion coefficients, and calibration image dimensions.
    """
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard[1], 0:checkerboard[0]].T.reshape(-1, 2)

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        return None

    frame_count = 0
    calibration_dim = None
    detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                detected_count += 1
                calibration_dim = gray.shape[::-1]
                objpoints.append(objp.copy())  # Ensure a new instance of objp is added
                corners2 = cv2.cornerSubPix(
                    gray, corners, (3, 3), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                )
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(frame, checkerboard, corners, ret)
                cv2.imshow("Checkerboard Detection", frame)
                cv2.waitKey(100)
            else:
                print(f"Checkerboard not found in frame {frame_count}.")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        print("Insufficient valid detections for calibration. Need at least 10.")
        return None

    print(f"Detected checkerboard in {detected_count} frames.")
    print(f"Calibration dimensions: {calibration_dim}")

    # Fisheye calibration
    N_imm = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_imm)]

    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

    try:
        _, K, D, _, _ = cv2.fisheye.calibrate(
            objpoints, imgpoints, calibration_dim, K, D, rvecs, tvecs, calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    except cv2.error as e:
        # Identify problematic frames and remove them
        problematic_frame_idx = int(str(e).split("input array ")[1].split(" ")[0])  # Extract index of problematic frame
        print(f"Problematic frame detected: Input array {problematic_frame_idx}. Removing it from calibration.")
        if problematic_frame_idx < len(objpoints):
            objpoints.pop(problematic_frame_idx)
            imgpoints.pop(problematic_frame_idx)

        # Retry calibration
        print("Retrying calibration after removing problematic frame...")
        try:
            _, K, D, _, _ = cv2.fisheye.calibrate(
                objpoints, imgpoints, calibration_dim, K, D, rvecs, tvecs, calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        except cv2.error as retry_error:
            print(f"Calibration failed again after removing problematic frame. Error: {retry_error}")
            return None

    print("Fisheye calibration completed successfully.")
    return K, D, calibration_dim


if __name__ == "__main__":

    calibration_result = calibrate_fisheye_video(video_file_path, checkerboard_size, frame_interval)
    if calibration_result is None:
        print("Calibration was unsuccessful. Please check the input video and checkerboard settings.")
    else:
        K, D, calibration_dim = calibration_result
        np.savez(calibration_output_path, K=K, D=D, calibration_dim=(calibration_dim[1], calibration_dim[0]))
        print("Calibration data saved to ",calibration_output_path)
        print("Saved calibration data with dimensions:", (calibration_dim[1], calibration_dim[0]))
