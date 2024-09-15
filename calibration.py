import numpy as np
from coin_detect import detect_yolo, detect_cv

image_path = ['images/tes3/baby_up.jpeg', 'images/tes3/baby_side.jpeg'] ### GANTI PATH INI

def calibration(img1, img2):
    COIN_COEFFICIENT1 = detect_yolo(image_path[0])
    COIN_COEFFICIENT2 = detect_yolo(image_path[1])
    #COIN_COEFFICIENT1 = detect_cv(image_path[0]) #komputasi lebih cepat
    #COIN_COEFFICIENT2 = detect_cv(image_path[1])
    coef = np.array([COIN_COEFFICIENT1, COIN_COEFFICIENT2])
    print(coef)

    #SIMPAN DALAM NPY
    np.save('coin_coeffs.npy', [COIN_COEFFICIENT1, COIN_COEFFICIENT2])
    #SIMPAN DALAM TXT
    np.savetxt('coin_coeffs.txt', coef, fmt='%.6f', header='COIN_COEFFS')

    return coef

calibration(image_path[0], image_path[1])



