import cv2
import matplotlib.pyplot as plt


img = cv2.imread('/home/samiksha/my_pycharm_projects/machine_learning_pycharm/Dataset_customized/images/000005.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_shape = img.shape
start_point = (5,5)
end_point = (100,100)
color = (255,0,0)
thickness = 2
image = cv2.rectangle(img,start_point,end_point,color,thickness)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite("exercise_4.png", image)

