import tensorflow as tf

# For downloading the image.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
import os

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def draw_boxes(img_path, obj_list, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  #font = ImageFont.load_default()
  im = np.array(Image.open(img_path), dtype=np.uint8)
  im_size = im.shape

  fig,ax = plt.subplots(1)
  ax.imshow(im)

  for obj in obj_list:
      ymin, xmin, ymax, xmax = tuple(obj_['bbox'])

      ymin *= im_size[0]
      xmin *= im_size[1]
      ymax = im_size[0]*ymax - ymin
      xmax = im_size[1]*xmax - xmin

      rect = patches.Rectangle((xmin,ymin),xmax,ymax,linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(rect)

  plt.show()


if __name__ == "__main__":
    import pickle
    with open('ucf_objects_detected_mobilenet.pickle', 'rb') as f:
        ucf_objects_mobilenet = pickle.load(f)
        
    img_name = '008552_2.jpg'
    img_path = os.path.join('/Users/katerina/Workspace/visual_census/ucf_data/part8', img_name)
    obj_list = ucf_objects_mobilenet[img_name]

    draw_boxes(img_path, obj_list, max_boxes=10, min_score=0.1)