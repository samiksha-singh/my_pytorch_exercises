import numpy as np
import cv2
from PIL import Image

f_name = "958.png"
a = cv2.imread(f_name)
a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
#a = a[200:1000 , 200:1000, :]
# a= cv2.resize(a, (200, 200))
b = np.zeros((300,300,3), dtype= np.uint8)
#b[:,:,:] = 255
h,w,c = a.shape
input_coord = (100,100)
b[input_coord[0]:input_coord[0]+h, input_coord[1]:input_coord[1]+w, :] = a # at b location a is being put
x = Image.fromarray(b) #numpy to PIL
x.show() #show the image
#x = x.save("cropped.png") #save file
