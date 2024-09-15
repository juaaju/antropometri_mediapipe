from utility import width_extraction, elips
from segmentation import get_input_point
import numpy as np

image_path = ['images/tes3/baby_up.jpeg', 'images/tes3/baby_side.jpeg'] ### GANTI PATH INI

#LOAOD HASIL KALIBRASI
COIN_COEFFICIENT1, COIN_COEFFICIENT2 = np.load('coin_coeffs.npy', allow_pickle=True)
print(COIN_COEFFICIENT1, COIN_COEFFICIENT2)

# MEDIAPIPE POSE DAN SEGMENTASI
def all_params(img1, img2):
  coords1, right_foot1, left_foot1, segmentation_mask1, top_head = get_input_point(img1)
  coords2, _, _, segmentation_mask2, _ = get_input_point(img2)

  total_badan = int(abs((top_head-coords1[1][0])+(coords1[1][0]-coords1[7][0]))*COIN_COEFFICIENT1+(right_foot1+left_foot1)*COIN_COEFFICIENT1/2)
  #print(top_head, right_foot1, left_foot1, coords1[1][0], coords1[7][0])

  half_chest_abs1 = abs(coords1[7][0]-coords1[1][0])/2
  chest_coordinate1 = int((coords1[1][0] + half_chest_abs1) * 0.95)
  abs_coordinates1 = int((coords1[1][0] + half_chest_abs1) * 1.15)
  #print(chest_coordinate1, abs_coordinates1)
  half_chest_abs2 = abs(coords2[7][0]-coords2[1][0])/2
  chest_coordinate2 = int((coords2[1][0]+half_chest_abs2)*0.95)
  abs_coordinates2 = int((coords2[1][0]+half_chest_abs2)*1.05)
  #print(half_chest_abs2,chest_coordinate2, abs_coordinates2)

  #upper camera
  head_w1 = width_extraction(segmentation_mask1, coords1[0][0]*0.9, coords1[0][1])
  #print('head_upper', head_w1)
  chest_w1 = width_extraction(segmentation_mask1, chest_coordinate1, coords1[0][1])
  abdomen_w1 = width_extraction(segmentation_mask1, abs_coordinates1, coords1[0][1])
  hand_w1 = width_extraction(segmentation_mask1, coords1[3][0], coords1[3][1])
  leg_w1 = width_extraction(segmentation_mask1, coords1[9][0], coords1[9][1])

  # coords[0][1] artinya sumbu y pada hidung menjadi acuan y

  #side camera
  head_w2 = width_extraction(segmentation_mask2, coords2[0][0]*0.9, coords2[0][1])
  #print('head_side', head_w2)
  chest_w2 = width_extraction(segmentation_mask2, chest_coordinate2, coords2[1][1])
  #print('chest_side', chest_w2)
  abdomen_w2 = width_extraction(segmentation_mask2, abs_coordinates2, coords2[1][1])
  hand_w2 = width_extraction(segmentation_mask2, coords2[3][0], coords2[3][1])
  leg_w2 = width_extraction(segmentation_mask2, coords2[9][0], coords2[9][1])

  #result
  head = round(elips(head_w1*COIN_COEFFICIENT1, head_w2*COIN_COEFFICIENT2))
  chest = round(elips(chest_w1*COIN_COEFFICIENT1, chest_w2*COIN_COEFFICIENT2))
  abdomen = round(elips(abdomen_w1*COIN_COEFFICIENT1, abdomen_w2*COIN_COEFFICIENT2))
  hand = round(elips(hand_w1*COIN_COEFFICIENT1, hand_w1*COIN_COEFFICIENT2*1.7))
  leg = round(elips(leg_w1*COIN_COEFFICIENT1, leg_w1*COIN_COEFFICIENT2*0.7))

  return head, chest, abdomen, hand, leg, total_badan

print(all_params(image_path[0], image_path[1]))
