import os
import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import walk, mkdir
import pandas as pd
from scipy.signal import convolve2d
import datetime

from random import randint

from image_corr_funcs import correlate_line, sweep_around, sweep_horizontal, find_depth, convert_depth_to_td_map, bound

# SETTINGS
plot_images = False
save_images = True
selected_images = range(105,140)  # 'all' or specific [###, ###, ..., ###] or range(start, stop, step)
# test_center_point = (120, 255)
test_kernel_size = (8, 8)
sweep_resolution = 10
line_resolution = 5
search_distance = 20
memory_loss = 0.5

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
    for image_out in ["original", "mixed", "squares", "depth", "combined",
                      "confidence", "hor", "memory", "td_radial", "td_hor"]:
        mkdir(f"{current_render_folder}/{image_out}")
else:
    current_render_folder = None


memory_map = np.zeros_like(images[0]['img'])


if selected_images == 'all':
    selected_images = range(len(images) - 1)
for current_image in selected_images:
    print(f"\033[1m=== Current Image: {current_image} ===\033[0m")
    current_image_name = images[current_image]['name'].removesuffix('.jpg') + '.png'

    if save_images:
        plt.imsave(f"{current_render_folder}/original/{current_image_name}", images[current_image]['img'].astype(np.uint8))


    # == Find corresponding data line ==
    time_index = dataframe['time'].sub(images[current_image]['time']).abs().idxmin()

    # (phi/roll, theta/pitch, psi/heading)
    attitude = (dataframe.iloc[time_index]['att_phi'],
                -dataframe.iloc[time_index]['att_theta'],
                dataframe.iloc[time_index]['att_psi'])

    print(f"\n=== ATTITUDE ===\n"
          f"\tphi   | roll:    {attitude[0] / np.pi * 180: 3.2f} [deg]\n"
          f"\ttheta | pitch:   {attitude[1] / np.pi * 180: 3.2f} [deg]\n"
          f"\tpsi   | heading: {attitude[2] / np.pi * 180: 3.2f} [deg]")

    # (x_world, y_world, z_world)
    vel_world = (dataframe.iloc[time_index]['vel_x'],
                 dataframe.iloc[time_index]['vel_y'],
                 dataframe.iloc[time_index]['vel_z'])

    R_phi = np.array([
        [1,                   0,                    0],
        [0, np.cos(attitude[0]), -np.sin(attitude[0])],
        [0, np.sin(attitude[0]),  np.cos(attitude[0])]])

    R_theta = np.array([
        [ np.cos(attitude[1]), 0, np.sin(attitude[1])],
        [                  0,  1,                   0],
        [-np.sin(attitude[1]), 0, np.cos(attitude[1])]])


    R_psi = np.array([
        [np.cos(attitude[2]), -np.sin(attitude[2]), 0],
        [np.sin(attitude[2]),  np.cos(attitude[2]), 0],
        [                  0,                    0, 1]])


    rotation_world_to_body = R_psi @ R_theta @ R_phi

    vel_body = rotation_world_to_body @ vel_world

    # TODO Find relation between body and camera frame
    vel_cam = vel_body

    vel_screen = vel_body / vel_body[1]

    print(f"\n=== VELOCITIES ===\n"
          f"\tWorld Velocity:  [{vel_world[0]: 1.2f}, {vel_world[1]: 1.2f}, {vel_world[2]: 1.2f}] (x,y,z) [m/s]\n"
          f"\tBody velocity:   [{vel_body[0]: 1.2f}, {vel_body[1]: 1.2f}, {vel_body[2]: 1.2f}] (x,y,z) [m/s]\n"
          f"\tCamera velocity: [{vel_body[0]: 1.2f}, {vel_body[1]: 1.2f}, {vel_body[2]: 1.2f}] (x,y,z) [m/s]  <- TODO\n"
          f"\tScreen velocity: [{vel_screen[0]: 1.2f}, {vel_screen[1]: 1.2f}, {vel_screen[2]: 1.2f}] (x,y,z) [K/x]")

    center_point = (bound(int(120 * (1 - vel_screen[0])), 10, 230),
                    bound(int(260 * (1 + vel_screen[2])), 10, 510))
    # center_point = (randint(10, 230),
    #                 randint(10, 510))

    if plot_images or save_images:
        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.imshow(images[current_image]['img'], alpha=0.5)
        plt.imshow(images[current_image + 1]['img'], alpha=0.5)
        plt.plot(center_point[1], center_point[0], 'rx')

    if plot_images:
        plt.title(str(images[10]['time']) + " + " + str(images[11]['time']))
        plt.show()

    if save_images:
        plt.axis('off')
        plt.savefig(f"{current_render_folder}/mixed/{current_image_name}", bbox_inches='tight', pad_inches=0, dpi=68)
        plt.close()

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

    # blur_kernel = 1/273 * np.array([[1,  4,  7,  4, 1],
    #                                 [4, 16, 26, 16, 4],
    #                                 [7, 26, 41, 26, 7],
    #                                 [4, 16, 26, 16, 4],
    #                                 [1,  4,  7,  4, 1]])
    # edge_kernel = np.array([[-1, 0, 1],
    #                         [-2, 0, 2],
    #                         [-1, 0, 1]])
    #
    # edge_map = np.mean(images[current_image + 1]['img'].astype(np.uint8), axis=2)
    # # for i in range(4):
    # edge_map = convolve2d(edge_map, edge_kernel, mode='same')
    # edge_map = np.absolute(edge_map)
    # edge_map = np.expand_dims(edge_map, axis=2)
    # edge_map = np.repeat(edge_map, 3, axis=2)
    #
    # prev_edge_map = np.mean(images[current_image]['img'].astype(np.uint8), axis=2)
    # # for i in range(4):
    # prev_edge_map = convolve2d(prev_edge_map, edge_kernel, mode='same')
    # prev_edge_map = np.absolute(prev_edge_map)
    # prev_edge_map = np.expand_dims(prev_edge_map, axis=2)
    # prev_edge_map = np.repeat(prev_edge_map, 3, axis=2)

    # == Sweep line around image ==
    if plot_images or save_images:
        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.imshow(images[current_image]['img'], alpha=0.5)
        plt.imshow(images[current_image + 1]['img'], alpha=0.5)


    sweep_result = sweep_around(center_point, images[current_image + 1]['img'], images[current_image]['img'],
                                test_kernel_size, sweep_resolution=sweep_resolution,
                                line_resolution=line_resolution, search_distance=search_distance,
                                plot=plot_images or save_images)

    hor_line_result = sweep_horizontal(edge_map, prev_edge_map,
                                       inter_line_distance=sweep_resolution,
                                       line_resolution=line_resolution, search_distance=search_distance,
                                       kernel_size=test_kernel_size, plot=plot_images or save_images)

    if plot_images:
        plt.title(str(images[10]['time']) + " + " + str(images[11]['time']))
        plt.show()
    if save_images:
        plt.axis('off')
        plt.savefig(f"{current_render_folder}/squares/{current_image_name}", bbox_inches='tight', pad_inches=0, dpi=68)
        plt.close()

    # == Show sample lines ==
    # if plot_images:
    #     for line in [0, 5, 10, 15]:
    #         fig = plt.figure(figsize=(6, 6), layout='constrained')
    #         plt.title(f"Line Correlation result of Line {line}")
    #         plt.imshow(sweep_result[line][0])
    #         plt.show()

    # == Find depth map ==
    depth_map = np.zeros_like(images[current_image]['img'])
    hor_depth_map = np.zeros_like(images[current_image]['img'])
    confidence_map = np.zeros_like(images[current_image]['img'])
    hor_confidence_map = np.zeros_like(images[current_image]['img'])

    for line in sweep_result:
        depth_map, confidence_map = find_depth(line, depth_map, confidence_map)

    for line in hor_line_result:
        hor_depth_map, hor_confidence_map = find_depth(line, hor_depth_map, hor_confidence_map)

    combined_map = (depth_map * confidence_map / 255).astype(np.uint8)
    hor_combined_map = (hor_depth_map * hor_confidence_map / 255).astype(np.uint8)

    # blur_kernel = 1/273 * np.array([[1,  4,  7,  4, 1],
    #                                 [4, 16, 26, 16, 4],
    #                                 [7, 26, 41, 26, 7],
    #                                 [4, 16, 26, 16, 4],
    #                                 [1,  4,  7,  4, 1]])
    # blur_kernel = np.array([[-1, 0, 1],
    #                         [-2, 0, 2],
    #                         [-1, 0, 1]])
    #
    # blur_map = np.mean(images[current_image]['img'].astype(np.uint8), axis=2)
    # # for i in range(4):
    # blur_map = convolve2d(blur_map, blur_kernel, mode='same')
    # blur_map = np.absolute(blur_map)
    # blur_map = np.expand_dims(blur_map, axis=2)
    # blur_map = np.repeat(blur_map, 3, axis=2)

    memory_map -= (memory_map * memory_loss).astype(np.uint8)
    memory_map += depth_map
    memory_map += hor_depth_map

    td_map = convert_depth_to_td_map(depth_map)
    hor_td_map = convert_depth_to_td_map(hor_depth_map)

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
        plt.title(f"Horizontal Line Depth Map")
        plt.imshow(hor_depth_map)
        plt.show()

        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Horizontal Line Confidence Map")
        plt.imshow(hor_confidence_map)
        plt.show()

        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Depth * Confidence Map")
        plt.imshow(combined_map)
        plt.show()

        # fig = plt.figure(figsize=(8, 4), layout='constrained')
        # plt.title(f"Blurred Map")
        # plt.imshow(blur_map)
        # plt.show()

        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Memory Map")
        plt.imshow(memory_map)
        plt.show()

        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Top-Down Map")
        plt.imshow(td_map)
        plt.show()

        fig = plt.figure(figsize=(8, 4), layout='constrained')
        plt.title(f"Top-Down (Hor) Map")
        plt.imshow(hor_td_map)
        plt.show()

    if save_images:
        plt.imsave(f"{current_render_folder}/depth/{current_image_name}", depth_map.astype(np.uint8))
        plt.imsave(f"{current_render_folder}/confidence/{current_image_name}", confidence_map.astype(np.uint8))
        plt.imsave(f"{current_render_folder}/combined/{current_image_name}", combined_map.astype(np.float64))
        plt.imsave(f"{current_render_folder}/hor/{current_image_name}", hor_depth_map.astype(np.uint8))
        plt.imsave(f"{current_render_folder}/memory/{current_image_name}", memory_map.astype(np.float64) / 256)
        plt.imsave(f"{current_render_folder}/td_radial/{current_image_name}", td_map.astype(np.uint8))
        plt.imsave(f"{current_render_folder}/td_hor/{current_image_name}", hor_td_map.astype(np.uint8))


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
