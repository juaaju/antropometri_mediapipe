from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math

# utils
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def width_extraction(mask, target_x, target_y):
    """
    Cek apakah di atas dan di bawah target_y pada target_x ada nilai 255 secara terus menerus
    sampai menemukan nilai yang tidak 255.

    Args:
    - mask: gambar hitam putih sebagai array numpy
    - target_x: posisi x yang ingin diperiksa
    - target_y: posisi y awal yang ingin diperiksa

    Returns:
    - min_y: nilai y terendah yang memiliki nilai 255
    - max_y: nilai y tertinggi yang memiliki nilai 255
    """
    target_x = int(target_x)
    target_y = int(target_y)

    # Ubah nilai 1 menjadi 255 di dalam mask
    mask[mask == 1] = 255

    # Dapatkan ukuran gambar
    height, width = mask.shape

    # Pastikan target_x berada dalam rentang gambar
    if target_x < 0 or target_x >= width:
        print("Koordinat x di luar rentang gambar.")
        return None

    # Mulai dari target_y, cari ke atas dan ke bawah
    min_y = target_y
    max_y = target_y

    # Cek ke bawah (y bertambah)
    for y in range(target_y, height):
        if mask[y, target_x] == 255:  # Ubah image ke mask
            max_y = y
        else:
            #print(f"Berhenti ke bawah pada y = {y}")
            break  # Berhenti jika tidak ada nilai 255 lagi

    # Cek ke atas (y berkurang)
    for y in range(target_y, -1, -1):
        if mask[y, target_x] == 255:  # Ubah image ke mask
            min_y = y
        else:
            #print(f"Berhenti ke atas pada y = {y}")
            break  # Berhenti jika tidak ada nilai 255 lagi

    width = abs(max_y - min_y)
    width = int(width)

    return width

def elips(sb_mayor, sb_minor):
  #hitung keliling elips
  elips_circumtances=0.5 * math.pi *(sb_mayor + sb_minor)
  return elips_circumtances

def find_first_one(image, y):
    # Memeriksa apakah y ada dalam batas gambar
    if y < 0 or y >= image.shape[0]:
        return None  # Mengembalikan None jika y di luar batas

    # Mengambil baris sesuai nilai y yang diinput
    row = image[y]
    
    # Mencari indeks pertama kemunculan nilai 1 di baris tersebut
    for i, value in enumerate(row):
        if value == 1:
            return i  # Mengembalikan indeks x ketika menemukan 1
    
    return None  # Mengembalikan None jika tidak ada nilai 1 di baris
