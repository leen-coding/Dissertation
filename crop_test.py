
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
Path = "D:\\Files\\arcface-pytorch\\mfr2\\BrianKemp\\BrianKemp_0004.png"
img = Image.open(Path)
input_size = (112,224)
img = img.resize(input_size)
ratio_down = 0.5
upper_size = int(input_size[1]*(1-ratio_down))
crop_img_up = img.crop((0,0,input_size[0],upper_size))
crop_img_down = img.crop((0,upper_size,input_size[0],input_size[1]))
plt.figure("1")
plt.imshow(crop_img_down)
plt.figure("2")
plt.imshow(crop_img_up)
plt.show()
