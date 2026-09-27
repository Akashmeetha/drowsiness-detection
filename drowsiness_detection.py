"""
Real-Time Driver Drowsiness & Yawning Detection System (Enhanced)
------------------------------------------------------------------
Uses:
  - OpenCV        -> Webcam capture + drawing + head-pose (solvePnP)
  - MediaPipe     -> Face Landmarker (Tasks API, 478 landmark points)
  - NumPy / SciPy -> Distance & angle calculations
  - simpleaudio   -> Alarm sound (optional; falls back to a terminal beep
                     if simpleaudio isn't installed/working on this PC)
  - csv           -> Event logging for later analysis / report

NOTE ON MEDIAPIPE VERSION:
  Newer MediaPipe releases (used automatically on newer Python versions
  like 3.13/3.14) removed the old `mp.solutions.face_mesh` API and
  replaced it with the new "Tasks" API (`mediapipe.tasks.python.vision`).
  This script uses the new Tasks API, so it works on both old and new
  MediaPipe installations. The first run downloads a small model file
  (`face_landmarker.task`, ~ a few MB) from Google -- this needs an
  internet connection ONLY the first time; after that it's cached
  locally next to this script.

FEATURES:
  1. EAR (Eye Aspect Ratio)   -> detects closed eyes (drowsiness)
  2. MAR (Mouth Aspect Ratio) -> detects yawning
  3. HEAD-DROOP DETECTION     -> cv2.solvePnP head pitch angle, detects
                                  the head nodding/dropping forward
  4. EVENT LOGGING (CSV)      -> every alert is timestamped in
                                  `session_log.csv`
  5. SESSION REPORT           -> a summary report (`session_report.txt`)
                                  is generated when you quit (press 'q')
"""

import os
import cv2
import csv
import time
import threading
import datetime
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist

# ------------------------------------------------------------------
# 1. CONFIGURATION / THRESHOLDS  (tune these based on testing)
# ------------------------------------------------------------------
EAR_THRESHOLD = 0.23           # Below this -> eyes considered "closed"
EAR_CONSEC_FRAMES = 20          # Consecutive frames closed -> drowsy

MAR_THRESHOLD = 0.6             # Above this -> mouth considered "open" (yawn)
MAR_CONSEC_FRAMES = 15          # Consecutive frames open -> yawn confirmed

PITCH_DROP_THRESHOLD = 15.0     # Degrees the head must dip forward/down
PITCH_CONSEC_FRAMES = 15        # Consecutive frames head must stay dropped

LOG_FILE = "session_log.csv"
REPORT_FILE = "session_report.txt"

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Landmark indices (same 468/478-point face mesh topology as before)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

POSE_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)


# ------------------------------------------------------------------
# 2. MODEL DOWNLOAD (one-time) -- NEW
# ------------------------------------------------------------------
def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print("First-time setup: downloading face landmark model (a few MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download the required model file. "
            f"Please check your internet connection and try again. ({e})"
        )


# ------------------------------------------------------------------
# 3. ALARM SOUND
# ------------------------------------------------------------------
def generate_beep(frequency=1000, duration=0.6, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = (tone * 32767 / np.max(np.abs(tone))).astype(np.int16)
    return audio, sample_rate


_alarm_playing = False


def play_alarm():
    global _alarm_playing
    if _alarm_playing:
        return
    _alarm_playing = True

    def _play():
        global _alarm_playing
        try:
            import simpleaudio as sa
            audio, sr = generate_beep(1000, 0.7)
            play_obj = sa.play_buffer(audio, 1, 2, sr)
            play_obj.wait_done()
        except Exception as e:
            print("\a", end="", flush=True)
            print(f"[Alarm sound fallback used] ({e})")
        finally:
            _alarm_playing = False

    threading.Thread(target=_play, daemon=True).start()


# ------------------------------------------------------------------
# 4. EAR / MAR CALCULATIONS
# ------------------------------------------------------------------
def eye_aspect_ratio(landmarks, eye_points, w, h):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
    p1, p2, p3, p4, p5, p6 = coords
    vertical1 = dist.euclidean(p2, p6)
    vertical2 = dist.euclidean(p3, p5)
    horizontal = dist.euclidean(p1, p4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear, coords


def mouth_aspect_ratio(landmarks, mouth_points, w, h):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_points]
    left, right, top1, bottom1, top2, bottom2, top3, bottom3 = coords
    horizontal = dist.euclidean(left, right)
    vertical = (dist.euclidean(top1, bottom1) +
                dist.euclidean(top2, bottom2) +
                dist.euclidean(top3, bottom3)) / 3.0
    mar = vertical / horizontal
    return mar, coords


# ------------------------------------------------------------------
# 5. HEAD-POSE (PITCH) ESTIMATION
# ------------------------------------------------------------------
def get_head_pitch(landmarks, w, h):
    image_points = np.array([
        (landmarks[POSE_LANDMARK_IDS["nose_tip"]].x * w, landmarks[POSE_LANDMARK_IDS["nose_tip"]].y * h),
        (landmarks[POSE_LANDMARK_IDS["chin"]].x * w, landmarks[POSE_LANDMARK_IDS["chin"]].y * h),
        (landmarks[POSE_LANDMARK_IDS["left_eye_corner"]].x * w, landmarks[POSE_LANDMARK_IDS["left_eye_corner"]].y * h),
        (landmarks[POSE_LANDMARK_IDS["right_eye_corner"]].x * w, landmarks[POSE_LANDMARK_IDS["right_eye_corner"]].y * h),
        (landmarks[POSE_LANDMARK_IDS["left_mouth_corner"]].x * w, landmarks[POSE_LANDMARK_IDS["left_mouth_corner"]].y * h),
        (landmarks[POSE_LANDMARK_IDS["right_mouth_corner"]].x * w, landmarks[POSE_LANDMARK_IDS["right_mouth_corner"]].y * h),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj_matrix = np.hstack((rotation_mat, translation_vec))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    # NOTE: euler_angles has shape (3,1). Newer NumPy (2.x) no longer allows
    # float() on a size-1 array directly, so we pull out the scalar with
    # .item() / [0][0] instead of float(euler_angles[0]).
    pitch = float(np.ravel(euler_angles)[0])

    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
    return pitch


# ------------------------------------------------------------------
# 6. LOGGING HELPERS
# ------------------------------------------------------------------
def init_log_file():
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event_type", "ear", "mar", "pitch_deg"])


def log_event(event_type, ear, mar, pitch):
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type, f"{ear:.3f}", f"{mar:.3f}", f"{pitch:.2f}"
        ])


def write_session_report(start_time, counts):
    duration_sec = time.time() - start_time
    minutes, seconds = divmod(int(duration_sec), 60)

    total_alerts = counts["drowsy"] + counts["yawn"] + counts["head_drop"]
    fatigue_score = counts["drowsy"] * 3 + counts["head_drop"] * 3 + counts["yawn"] * 1

    with open(REPORT_FILE, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("  DRIVER DROWSINESS - SESSION REPORT\n")
        f.write("=" * 55 + "\n")
        f.write(f"Session Duration      : {minutes} min {seconds} sec\n")
        f.write(f"Drowsiness Alerts      : {counts['drowsy']}\n")
        f.write(f"Yawn Alerts            : {counts['yawn']}\n")
        f.write(f"Head-Drop Alerts       : {counts['head_drop']}\n")
        f.write(f"Total Alert Events     : {total_alerts}\n")
        f.write(f"Fatigue Score          : {fatigue_score}\n")
        if fatigue_score == 0:
            verdict = "Driver appeared ALERT throughout the session."
        elif fatigue_score <= 5:
            verdict = "Mild fatigue signs detected. Recommend a short break soon."
        elif fatigue_score <= 15:
            verdict = "Moderate fatigue detected. A break is recommended."
        else:
            verdict = "HIGH fatigue risk detected. Driver should stop and rest immediately."
        f.write(f"Verdict                : {verdict}\n")
        f.write("=" * 55 + "\n")
        f.write(f"Detailed event log saved in: {LOG_FILE}\n")

    print("\n" + open(REPORT_FILE).read())


# ------------------------------------------------------------------
# 7. MAIN DETECTION LOOP
# ------------------------------------------------------------------
def main():
    ensure_model_downloaded()

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam open aagala. Camera index check pannunga.")
        return

    init_log_file()
    session_start = time.time()
    event_counts = {"drowsy": 0, "yawn": 0, "head_drop": 0}

    eye_counter = 0
    yawn_counter = 0
    pitch_counter = 0

    drowsy_active = False
    yawn_active = False
    head_drop_active = False

    prev_time = time.time()
    print("Drowsiness detection started... 'q' press pannina exit aagum.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                left_ear, left_coords = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear, right_coords = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                mar, mouth_coords = mouth_aspect_ratio(landmarks, MOUTH, w, h)
                pitch = get_head_pitch(landmarks, w, h)

                for (x, y) in left_coords + right_coords:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                for (x, y) in mouth_coords:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                # ---- Drowsiness (EAR) ----
                if ear < EAR_THRESHOLD:
                    eye_counter += 1
                    if eye_counter >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        play_alarm()
                        if not drowsy_active:
                            event_counts["drowsy"] += 1
                            log_event("DROWSY", ear, mar, pitch)
                        drowsy_active = True
                else:
                    eye_counter = 0
                    drowsy_active = False

                # ---- Yawning (MAR) ----
                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= MAR_CONSEC_FRAMES:
                        cv2.putText(frame, "YAWN ALERT!", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 3)
                        play_alarm()
                        if not yawn_active:
                            event_counts["yawn"] += 1
                            log_event("YAWN", ear, mar, pitch)
                        yawn_active = True
                else:
                    yawn_counter = 0
                    yawn_active = False

                # ---- Head-Drop ----
                if pitch > PITCH_DROP_THRESHOLD:
                    pitch_counter += 1
                    if pitch_counter >= PITCH_CONSEC_FRAMES:
                        cv2.putText(frame, "HEAD-DROP ALERT!", (30, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        play_alarm()
                        if not head_drop_active:
                            event_counts["head_drop"] += 1
                            log_event("HEAD_DROP", ear, mar, pitch)
                        head_drop_active = True
                else:
                    pitch_counter = 0
                    head_drop_active = False

                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (30, 215),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Face not detected", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Driver Drowsiness & Yawning Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        write_session_report(session_start, event_counts)


if __name__ == "__main__":
    main()
