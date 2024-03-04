from PIL import Image
import numpy as np
import scipy as sp

image_1 = Image.open("../data/datasets/cyberzoo_aggressive_flight/20190121-144646/34581884.jpg")
image_2 = Image.open("../data/datasets/cyberzoo_aggressive_flight/20190121-144646/34681887.jpg")

image_array_1 = np.array(image_1)[:, :, 0]
image_array_2 = np.array(image_2)[:, :, 0]

print(image_array_1.shape)

test_kernel = np.array([[0.05, 0.15, 0.05],
                        [0.15, 0.2, 0.15],
                        [0.05, 0.15, 0.05]])

# image_array_3 = sp.signal.convolve2d(test_kernel, image_array_2)
#
# image_3 = Image.fromarray(image_array_3)
# image_3.show()
