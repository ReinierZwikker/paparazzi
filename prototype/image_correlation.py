import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import walk, mkdir
import pandas as pd
from scipy.signal import convolve2d
import datetime

from image_corr_funcs import correlate_line, sweep_around, find_depth

# SETTINGS
plot_images = False
save_images = True
selected_images = 'all'  # 'all' or specific [###, ###, ..., ###]
test_center_point = (120, 255)
test_kernel_size = (4, 4)

# == Loading dataset ==
file_names = []
for _, _, file_names in walk("../data/dataset/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935"):
    break

images = []
for file_name in file_names:
    with Image.open(f"../data/dataset/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/{file_name}") as image_file:
        images.append({'img': np.array(image_file.rotate(90, expand=True)), 'time': float(file_name.removesuffix('.jpg'))/(10**6), 'name': file_name})

images = sorted(images, key=lambda x: x['time'])

image_size = images[0]['img'].shape

with open("../data/dataset/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142943.csv") as csv_file:
    dataframe = pd.read_csv(csv_file)

if save_images:
    current_render_folder = f"../data/renders/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mkdir(f"{current_render_folder}")
    for image_out in ["original", "mixed", "squares", "depth", "confidence", "combined", "blur"]:
        mkdir(f"{current_render_folder}/{image_out}")
else:
    current_render_folder = None


if selected_images == 'all':
    selected_images = range(len(images) - 1)
for current_image in selected_images:
    print(f"\033[1m=== Current Image: {current_image} ===\033[0m")
    current_image_name = images[current_image]['name'].removesuffix('.jpg') + '.png'

    # == Find corresponding data line ==
    time_index = dataframe['time'].sub(images[current_image]['time']).abs().idxmin()

    # (phi/roll, theta/pitch, psi/heading)
    attitude = (dataframe.iloc[time_index]['att_phi'],
                dataframe.iloc[time_index]['att_theta'],
                dataframe.iloc[time_index]['att_psi'])

    # (x_world, y_world, z_world)
    vel_world = (dataframe.iloc[time_index]['vel_x'],
                 dataframe.iloc[time_index]['vel_y'],
                 dataframe.iloc[time_index]['vel_z'])

    R_phi = np.array([
        [np.cos(attitude[0]), -np.sin(attitude[0]), 0],
        [np.sin(attitude[0]),  np.cos(attitude[0]), 0],
        [                  0,                    0, 1]])

    R_theta = np.array([
        [ np.cos(attitude[1]), 0, np.sin(attitude[1])],
        [                  0,  1,                   0],
        [-np.sin(attitude[1]), 0, np.cos(attitude[1])]])


    R_psi = np.array([
        [1,                   0,                    0],
        [0, np.cos(attitude[2]), -np.sin(attitude[2])],
        [0, np.sin(attitude[2]),  np.cos(attitude[2])]])

    rotation_world_to_body = R_psi @ R_theta @ R_phi

    vel_body = rotation_world_to_body @ vel_world

    # TODO Find relation between body and camera frame
    vel_cam = vel_body

    vel_screen = vel_body / vel_body[0]
    print(f"\n=== VELOCITIES ===\n"
          f"\tWorld Velocity:  {vel_world}\n"
          f"\tBody velocity:   {vel_body}\n"
          f"\tCamera velocity: {vel_body}\n"
          f"\tScreen velocity: {vel_screen}")

    if plot_images:
        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(str(images[10]['time']) + " + " + str(images[11]['time']))
        plt.imshow(images[current_image]['img'], alpha=0.5)
        plt.imshow(images[current_image + 1]['img'], alpha=0.5)
        plt.show()

    if save_images:
        Image.fromarray(images[current_image]['img'], mode='RGB').save(f"{current_render_folder}/original/{current_image_name}")
        plt.imsave(f"{current_render_folder}/mixed/{current_image_name}",
                   (images[current_image]['img'] * 0.5 + images[current_image + 1]['img'] * 0.5).astype(np.uint8))

    # == Test single line ==
    # TEST CORRELATE LINE
    # line_start = (15, 15)
    # line_end = (125, 300)
    # kernel_size = (4, 4)
    #
    # line_result = correlate_line(images[current_image + 1]['img'], images[current_image]['img'], kernel_size,
    #                              line_start[0], line_start[1], line_end[0], line_end[1], step=2, logarithmic_spacing=False)
    # plt.show()
    #
    # fig = plt.figure(figsize=(6, 6), layout='constrained')
    # plt.title(f"Line Correlation result from {line_start} to {line_end}")
    # plt.imshow(line_result[0])
    # plt.show()

    # == Sweep line around image ==
    if plot_images or save_images:
        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.imshow(images[current_image]['img'], alpha=0.5)
        plt.imshow(images[current_image + 1]['img'], alpha=0.5)

    sweep_result = sweep_around(test_center_point, images[current_image + 1]['img'], images[current_image]['img'],
                                test_kernel_size, sweep_resolution=20, line_resolution=10,
                                plot=plot_images or save_images)

    if plot_images:
        plt.title(str(images[10]['time']) + " + " + str(images[11]['time']))
        plt.show()
    if save_images:
        plt.axis('off')
        plt.savefig(f"{current_render_folder}/squares/{current_image_name}", bbox_inches='tight', pad_inches=0)

    # == Show sample lines ==
    # if plot_images:
    #     for line in [0, 10, 20, 30, 40, 50, 60]:
    #         fig = plt.figure(figsize=(6, 6), layout='constrained')
    #         plt.title(f"Line Correlation result of Line {line}")
    #         plt.imshow(sweep_result[line][0])
    #         plt.show()

    # == Find depth map ==
    depth_map = np.zeros_like(images[current_image]['img'])
    confidence_map = np.zeros_like(images[current_image]['img'])

    for line in sweep_result:
        depth_map, confidence_map = find_depth(line, depth_map, confidence_map)


    combined_map = depth_map * confidence_map / 255

    blur_kernel = 1/273 * np.array([[1,  4,  7,  4, 1],
                                    [4, 16, 26, 16, 4],
                                    [7, 26, 41, 26, 7],
                                    [4, 16, 26, 16, 4],
                                    [1,  4,  7,  4, 1]])
    blur_map = np.mean(depth_map * confidence_map * 255, axis=2)
    for i in range(100):
        blur_map = convolve2d(blur_map, blur_kernel, mode='same')

    if plot_images:
        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Depth Map")
        plt.imshow(depth_map)
        plt.show()

        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Confidence Map")
        plt.imshow(confidence_map)
        plt.show()

        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Depth * Confidence Map")
        plt.imshow(combined_map)
        plt.show()

        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Blurred Map")
        plt.imshow(blur_map)
        plt.show()

    if save_images:
        plt.imsave(f"{current_render_folder}/depth/{current_image_name}", depth_map.astype(np.uint8))
        plt.imsave(f"{current_render_folder}/confidence/{current_image_name}", confidence_map.astype(np.uint8))
        plt.imsave(f"{current_render_folder}/combined/{current_image_name}", combined_map.astype(np.float64))
        plt.imsave(f"{current_render_folder}/blur/{current_image_name}", blur_map.astype(np.float64) / 255)


# Correlation EXAMPLE OLD
# x = 100
# y = 480
# c = 0
# print(f"EXAMPLE: Correlation for {x}, {y}, {c}")
#
# rectangle = np.copy(images[current_image]['img'][x - rectangle_extend[0]:x + rectangle_extend[0],
#                     y - rectangle_extend[1]:y + rectangle_extend[1],
#                     c])
# rectangle = rectangle.astype('int16') - rectangle.mean().astype('int16')
#
# plt.imshow(rectangle)
# plt.show()
#
# likeness_image = correlate2d(
#     images[current_image + 1]['img'][:, :, c].astype('int16') - images[current_image + 1]['img'][:, :, c].mean().astype(
#         'int16'), rectangle, boundary='symm', mode='same')
#
# match_x, match_y = np.unravel_index(np.argmax(likeness_image), likeness_image.shape)
#
# print(f"x={match_x}, y={match_y}; certainty?: {(likeness_image[match_x, match_y] - likeness_image.mean()) / 100000:3.2f}")
#
# movement = (x - match_x,
#             y - match_y)
#
# print(f"Error={movement}")
#
# plt.imshow(likeness_image)
# plt.plot(match_y, match_x, 'ro')
# plt.plot(y, x, 'rx')
# plt.show()
#
#
# # MOVEMENT FIELD CALC
# movement_field_mag = np.zeros(image_size)
#
#
# for x in range(rectangle_extend[0], image_size[0] - rectangle_extend[0], rectangle_size[0]):
#     for y in range(rectangle_extend[1], image_size[1] - rectangle_extend[1], rectangle_size[0]):
#         for c in [0, 1, 2]:
#             print(f"Correlation for {x}, {y}, {c}")
#
#             rectangle = np.copy(images[current_image]['img'][x - rectangle_extend[0]:x + rectangle_extend[0],
#                                 y - rectangle_extend[1]:y + rectangle_extend[1],
#                                 c])
#             rectangle = rectangle.astype('int16') - rectangle.mean().astype('int16')
#
#             # plt.imshow(rectangle)
#             # plt.show()
#
#             likeness_image = correlate2d(images[current_image + 1]['img'][:, :, c].astype('int16') - images[current_image + 1]['img'][:, :, c].mean().astype('int16'), rectangle, boundary='symm', mode='same')
#
#             match_x, match_y = np.unravel_index(np.argmax(likeness_image), likeness_image.shape)
#
#             print(f"x={match_x}, y={match_y}; certainty?: {(likeness_image[match_x, match_y] - likeness_image.mean()) / 100000:3.2f}")
#
#             movement = (x - match_x,
#                         y - match_y)
#
#             print(f"Movement={movement}")
#
#             movement_field_mag[x, y] += (movement[0]**2 + movement[1]**2)**0.5
#
#             # plt.imshow(likeness_image)
#             # plt.plot(match_y, match_x, 'ro')
#             # plt.plot(y, x, 'rx')
#             # plt.show()
#
# movement_field_mag /= movement_field_mag.max()
#
# plt.imshow(movement_field_mag)
# plt.show()
