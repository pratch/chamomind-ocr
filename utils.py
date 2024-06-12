import cv2

def cv2_imshow_at_height(winname, img, height=900):
  h, w, _ = img.shape
  new_w = int(w*height/h)
  resized_img = cv2.resize(img, (new_w, height))
  cv2.imshow(winname, resized_img, height) 