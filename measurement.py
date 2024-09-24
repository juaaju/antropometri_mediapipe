from utility import width_extraction, elips
from segmentation import get_input_point
import numpy as np
import json
from datetime import datetime
from datetime import datetime
now = datetime.now()
formatted_date = now.strftime("%d %B %Y, %H:%M")

#image_path = ['images/captured_image(4).jpg', 'images/captured_image1(4).jpg'] ### GANTI PATH INI

def load_list_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(new_data, filename):
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["results"].append({
            "data": new_data,
            "date": formatted_date
        })
        
        file.seek(0)
        json.dump(file_data, file, indent=4)
        file.truncate()

# #LOAOD HASIL 
# f = open('coin_coeffs.json')
# COIN_COEFFICIENT1, COIN_COEFFICIENT2  = json.load(f)
# # COIN_COEFFICIENT1, COIN_COEFFICIENT2 = np.load('coin_coeffs.npy', allow_pickle=True)
# print(COIN_COEFFICIENT1, COIN_COEFFICIENT2)

# MEDIAPIPE POSE DAN SEGMENTASI
def measure(img1, img2):
  COIN_COEFFICIENT1, COIN_COEFFICIENT2 = load_list_from_json('coin_coeffs.json')
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
  head = round(elips(head_w1*COIN_COEFFICIENT1, head_w2*COIN_COEFFICIENT2)*1.15)
  chest = round(elips(chest_w1*COIN_COEFFICIENT1, chest_w2*COIN_COEFFICIENT2)*1.1)
  abdomen = round(elips(abdomen_w1*COIN_COEFFICIENT1, abdomen_w2*COIN_COEFFICIENT2)*1.1)
  hand = round(elips(hand_w1*COIN_COEFFICIENT1, hand_w1*COIN_COEFFICIENT2*1.7))
  leg = round(elips(leg_w1*COIN_COEFFICIENT1, leg_w1*COIN_COEFFICIENT2*0.7))

  data = {
    "head": head,
    "chest": chest,
    "abdomen": abdomen,
    "hand": hand,
    "leg": leg,
    "height": total_badan}
  
  result_filename = 'results.json'
  write_json(data, result_filename)
  
  return head, chest, abdomen, hand, leg, total_badan

# print(measure(image_path[0], image_path[1]))
