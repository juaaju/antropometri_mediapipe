import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

#coin detect using yolo
def detect_yolo(img):
  COIN_DIAMETER = 2.7 #cm
  model = torch.hub.load('.', 'custom', path='model/best.pt', source='local', force_reload=True)
  # Image
  # Inference
  results = model(img)

  # Results, change the flowing to: results.show()
  df = results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc
  df = df[df['confidence'] > 0.6]
  df['x'] = df['xmax'] - df['xmin']
  df['y'] = df['ymax'] - df['ymin']
  df['x_tengah'] = (df['xmin'] + df['xmax']) / 2
  df['y_tengah'] = (df['ymin'] + df['ymax']) / 2
  df = df.sort_values('x')
  df = df.reset_index()
  x = df['x_tengah'][0]
  y = df['y_tengah'][0]
  # print(df)
  coords = np.array([[x,y]])
  lists = [df['xmin'][0], df['ymin'][0], df['xmax'][0], df['ymax'][0]]
  width = lists[2] - lists[0]
  height = lists[3] - lists[1]
  if width > height:
      width = height
  else:
      pass
  
  coefficient = COIN_DIAMETER / width

  return coefficient

#print(detect('images/baby_up.jpeg'))

#coin detect using opencv Houghcircles
def detect_cv(image_path):
    COIN_DIAMETER = 2.7 #cm
    # Baca gambar
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Konversi gambar ke skala abu-abu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur gambar untuk mengurangi noise
    gray_blurred = cv2.medianBlur(gray, 5)

    # Deteksi lingkaran menggunakan HoughCircles

    circles = cv2.HoughCircles(gray_blurred,
               cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
           param2 = 30, minRadius = 1, maxRadius = 40)

    # Jika lingkaran terdeteksi
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Urutkan lingkaran berdasarkan radius (semakin besar, semakin "akurat")
        best_circle = max(circles, key=lambda c: c[2])
        x, y, r = best_circle

        # Gambar lingkaran dan pusatnya
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        # Konversi BGR ke RGB untuk plt.imshow
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Tampilkan gambar dengan lingkaran yang terdeteksi
        #plt.figure(figsize=(8, 8))
        #plt.imshow(img_rgb)
        #plt.axis("on")
        #plt.show()

        #coefficient coin
        coefficient = COIN_DIAMETER / (r * 2.0)

        # Kembalikan koordinat pusat lingkaran terbaik
        return coefficient
    else:
        print("Tidak ada lingkaran yang terdeteksi.")
        return None