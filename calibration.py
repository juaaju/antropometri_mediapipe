import numpy as np
from coin_detect import detect_yolo, detect_cv
import json

# image_path = ['images/captured_image(4).jpg', 'images/captured_image1(4).jpg'] ### GANTI PATH INI

def calibration(img1, img2):
    def save_to_json(data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file)
    COIN_COEFFICIENT1 = detect_yolo(img1)
    COIN_COEFFICIENT2 = detect_yolo(img2)
    #COIN_COEFFICIENT1 = detect_cv(img1) #komputasi lebih cepat
    #COIN_COEFFICIENT2 = detect_cv(img2)
    coef = np.array([COIN_COEFFICIENT1, COIN_COEFFICIENT2])
    print(coef)

    #SIMPAN DALAM NPY
    # np.save('coin_coeffs.npy', [COIN_COEFFICIENT1, COIN_COEFFICIENT2])
    # #SIMPAN DALAM TXT
    # np.savetxt('coin_coeffs.txt', coef, fmt='%.6f', header='COIN_COEFFS')
    
    #simpan dalam json
    save_to_json([COIN_COEFFICIENT1, COIN_COEFFICIENT2], 'coin_coeffs.json')
    print('tersimpan di json')

    return coef

# calibration(image_path[0], image_path[1])



