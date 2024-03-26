import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


path = r"../data/datasets/own_data/20240315-103300" # folder where images are
images_file_names = sorted(os.listdir(path))

def convert_image_data_to_yuv(image_data: np) -> np.array:
        m = np.array([[0.29900, -0.16874, 0.50000],
                      [0.58700, -0.33126, -0.41869],
                      [0.11400, 0.50000, -0.08131]])

        image_data_yuv = np.dot(image_data, m)
        image_data_yuv[:, :, 1:] += 127.0
        return image_data_yuv

def green_filter(image_data_yuv: np.array) -> np.array:

    pooled_image_data_yuv = np.zeros((image_data_yuv.shape[0] // kernel_size[0], image_data_yuv.shape[1] // kernel_size[1], image_data_yuv.shape[2]))

    for kernel_row in range(pooled_image_data_yuv.shape[0]):
        for kernel_col in range(pooled_image_data_yuv.shape[1]):
            for channel in range(pooled_image_data_yuv.shape[2]):
                pooled_image_data_yuv[kernel_row, kernel_col, channel] = np.mean(image_data_yuv[kernel_row*kernel_size[0]:(kernel_row+1)*kernel_size[0], kernel_col*kernel_size[1]:(kernel_col+1)*kernel_size[1], channel])

    filtered_image_data = np.zeros_like(pooled_image_data_yuv)
    filtered_image_data[:, :, 0][np.logical_and(pooled_image_data_yuv[:, :, 0] > y_min, pooled_image_data_yuv[:, :, 0] < y_max)] = True
    filtered_image_data[:, :, 1][np.logical_and(pooled_image_data_yuv[:, :, 1] > u_min, pooled_image_data_yuv[:, :, 1] < u_max)] = True
    filtered_image_data[:, :, 2][np.logical_and(pooled_image_data_yuv[:, :, 2] > v_min, pooled_image_data_yuv[:, :, 2] < v_max)] = True

    return filtered_image_data.all(axis=2)

def determine_direction(desired_heading, reduced_filtered_image):
    ray_scores = np.zeros((number_of_rays,), dtype=int)

    reduced_filtered_image_center_x = reduced_filtered_image.shape[1] // 2

    for row in range(reduced_filtered_image.shape[0]):
        for col in range(reduced_filtered_image.shape[1]):
            if reduced_filtered_image[row, col] == True:
                angle = np.arctan2((reduced_filtered_image.shape[0] - row) * kernel_size[0], (col - reduced_filtered_image_center_x) * kernel_size[1])
                ray_scores[np.argmin(np.abs(ray_angles - angle))] += 1

    desired_heading = lowpasfil_coefficient * ray_angles[np.argmax(ray_scores*ray_weighing)] + (1 - lowpasfil_coefficient) * desired_heading
    return desired_heading


# Parameters
y_min, y_max = 80, 255      # 80, 255 for 2023data; 50, 200 for 2019data
u_min, u_max = 30, 89       # 30, 89 for 2023data; 120, 135 for 2019data
v_min, v_max = 90, 145      # 90, 145 for 2023data; 80, 130 for 2019data
kernel_size = (24, 10)
number_of_rays = 7
ray_weighing = np.array([0.1, 0.5, 0.85, 1, 0.85, 0.5, 0.1])
lowpasfil_coefficient = 0.08


ray_angles = np.linspace(np.pi/6, 5*np.pi/6, number_of_rays)
desired_heading = np.pi/2

fig = plt.figure()
ax = fig.add_subplot(111)

raw_image_data = np.asarray(Image.open(os.path.join(path, images_file_names[0])).rotate(90, Image.NEAREST, expand = 1))
reduced_filtered_image_data = green_filter(convert_image_data_to_yuv(raw_image_data))
masked_image_data = raw_image_data * np.expand_dims(reduced_filtered_image_data.repeat(kernel_size[0], axis=0).repeat(kernel_size[1], axis=1), axis=2)

desired_heading = determine_direction(desired_heading, reduced_filtered_image_data)


image_plot = ax.imshow(masked_image_data.astype(int))
background_plot = ax.imshow(raw_image_data, alpha=0.4)
heading_plot, = ax.plot([masked_image_data.shape[1]/2, masked_image_data.shape[1]/2 + 100*np.cos(desired_heading)], [masked_image_data.shape[0], masked_image_data.shape[0] - 100*np.sin(desired_heading)],
                        linewidth=2,
                        color="magenta")

for image_file_name in images_file_names:
    raw_image_data = np.asarray(Image.open(os.path.join(path, image_file_name)).rotate(90, Image.NEAREST, expand = 1))
    reduced_filtered_image_data = green_filter(convert_image_data_to_yuv(raw_image_data))
    masked_image_data = raw_image_data * np.expand_dims(reduced_filtered_image_data.repeat(kernel_size[0], axis=0).repeat(kernel_size[1], axis=1), axis=2)

    desired_heading = determine_direction(desired_heading, reduced_filtered_image_data)

    image_plot.set_data(masked_image_data.astype(int))
    background_plot.set_data(raw_image_data)
    heading_plot.set_data([masked_image_data.shape[1]/2, masked_image_data.shape[1]/2 + 100*np.cos(desired_heading)], [masked_image_data.shape[0], masked_image_data.shape[0] - 100*np.sin(desired_heading)])
    plt.pause(0.3)

plt.show()
