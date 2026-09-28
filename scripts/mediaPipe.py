import glob
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(BASE_DIR, "data", "Videos")
MODELS_DIR = os.path.join(BASE_DIR, "models")
POSE_DIR = os.path.join(BASE_DIR, "output", "pose")
os.makedirs(POSE_DIR, exist_ok=True)

FPS = 30.0
MODELO = "heavy"        

NOMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

CAMPOS = ["x", "y", "z", "wx", "wy", "wz", "vis"]

COLUNAS = ["frame", "detectado"]
for nome in NOMES:
    for campo in CAMPOS:
        COLUNAS.append(f"{nome}_{campo}")


def ler_imagem(path):
    buf = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def achar_frames(seq_dir):
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    if frames:
        return frames
    for sub in sorted(glob.glob(os.path.join(seq_dir, "*"))):
        if os.path.isdir(sub):
            frames = sorted(glob.glob(os.path.join(sub, "*.png")))
            if frames:
                return frames
    return []


def montar_linha(frame, landmarks, world_landmarks):
    if landmarks is None:
        return [frame, 0] + [0.0] * (len(NOMES) * len(CAMPOS))
    
    linha = [frame, 1]

    for i in range(len(NOMES)):
        p = landmarks[i]
        w = world_landmarks[i]
        linha += [p.x, p.y, p.z, w.x, w.y, w.z, p.visibility]
    return linha


def processar_sequencia(seq_id, frame_paths, options):
    caminho_csv = os.path.join(POSE_DIR, f"{seq_id}.csv")
    n_sem_deteccao = 0

    with open(caminho_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUNAS)
        with vision.PoseLandmarker.create_from_options(options) as detector:
            for i, path in enumerate(frame_paths):
                imagem = ler_imagem(path)
                timestamp_ms = int(i * 1000 / FPS)
                resultado = detector.detect_for_video(imagem, timestamp_ms)
                frame = i + 1

                if resultado.pose_landmarks and resultado.pose_world_landmarks:
                    writer.writerow(montar_linha(frame,
                                                 resultado.pose_landmarks[0],
                                                 resultado.pose_world_landmarks[0]))
                else:
                    n_sem_deteccao += 1
                    writer.writerow(montar_linha(frame, None, None))

    return len(frame_paths), n_sem_deteccao


with open(os.path.join(MODELS_DIR, f"pose_landmarker_{MODELO}.task"), "rb") as f:
    base_options = python.BaseOptions(model_asset_buffer=f.read())

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

seq_dirs = sorted(d for d in glob.glob(os.path.join(VIDEOS_DIR, "*")) if os.path.isdir(d))
print(f"{len(seq_dirs)} sequencias | modelo {MODELO}\n")

for seq_dir in seq_dirs:
    seq_id = os.path.basename(seq_dir)
    frame_paths = achar_frames(seq_dir)

    if not frame_paths:
        print(f"  {seq_id:24s} sem PNG, pulada")
        continue

    n, sem = processar_sequencia(seq_id, frame_paths, options)
    print(f"  {seq_id:24s} {n:4d} frames | {sem} sem deteccao")

print(f"\nCSVs em {POSE_DIR}")
print("Agora rode: python scripts/juntar_rotulos.py")
