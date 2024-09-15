from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
from utility import calculate_distance, draw_landmarks_on_image, find_first_one

def get_input_point(image):
  base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_lite.task')
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)
  image = mp.Image.create_from_file(image)

  detection_result = detector.detect(image)
  pose_landmarks = detection_result.pose_landmarks

  nose = pose_landmarks[0][0]
  right_shoulder = pose_landmarks[0][12]
  right_elbow = pose_landmarks[0][14]
  right_wrist = pose_landmarks[0][16]
  left_shoulder = pose_landmarks[0][11]
  left_elbow = pose_landmarks[0][13]
  left_wrist = pose_landmarks[0][15]
  left_hip = pose_landmarks[0][23]
  left_knee = pose_landmarks[0][25]
  left_ankle = pose_landmarks[0][27]
  right_hip = pose_landmarks[0][24]
  right_knee = pose_landmarks[0][26]
  right_ankle = pose_landmarks[0][28]

  right_hand = calculate_distance(right_shoulder.x*image.width, right_shoulder.y*image.height, right_elbow.x*image.width, right_elbow.y*image.height) + calculate_distance(right_elbow.x*image.width, right_elbow.y*image.height, right_wrist.x*image.width, right_wrist.y*image.height)

  left_hand = calculate_distance(left_shoulder.x*image.width, left_shoulder.y*image.height, left_elbow.x*image.width, left_elbow.y*image.height) + calculate_distance(left_elbow.x*image.width, left_elbow.y*image.height, left_wrist.x*image.width, left_wrist.y*image.height)

  right_foot = calculate_distance(right_hip.x*image.width, right_hip.y*image.height, right_knee.x*image.width, right_knee.y*image.height) + calculate_distance(right_knee.x*image.width, right_knee.y*image.height, right_ankle.x*image.width, right_ankle.y*image.height)

  left_foot = calculate_distance(left_hip.x*image.width, left_hip.y*image.height, left_knee.x*image.width, left_knee.y*image.height) + calculate_distance(left_knee.x*image.width, left_knee.y*image.height, left_ankle.x*image.width, left_ankle.y*image.height)

  coords = np.array([
      [nose.x*image.width,nose.y*image.height],
      [right_shoulder.x*image.width,right_shoulder.y*image.height],
      [left_shoulder.x*image.width,left_shoulder.y*image.height],
      [right_elbow.x*image.width,right_elbow.y*image.height],
      [left_elbow.x*image.width,left_elbow.y*image.height],
      [right_wrist.x*image.width,right_wrist.y*image.height],
      [left_wrist.x*image.width,left_wrist.y*image.height],
      [right_hip.x*image.width,right_hip.y*image.height],
      [left_hip.x*image.width,left_hip.y*image.height],
      [right_knee.x*image.width,right_knee.y*image.height],
      [left_knee.x*image.width,left_knee.y*image.height],
      [right_ankle.x*image.width,right_ankle.y*image.height],
      [left_ankle.x*image.width,left_ankle.y*image.height],
      ])


  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
  segmentation_mask=segmentation_mask.astype(np.uint8)
  visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255

  # Display the annotated image in RGB format
  #plt.figure(figsize=(10, 10))
  #plt.imshow(annotated_image)
  #plt.title(f"Pose Result")
  #plt.show()

  # Display mask result
  #plt.figure(figsize=(10, 10))
  #plt.imshow(visualized_mask)
  #plt.title(f"Mask Result")
  #plt.show()

  #segment_coordinates = np.where(segmentation_mask == 1)
  #segment_x = segment_coordinates[1]
  #segment_y = segment_coordinates[0]
  top_head = find_first_one(segmentation_mask, int(coords[0][1]))

  return coords, right_foot, left_foot, segmentation_mask, top_head