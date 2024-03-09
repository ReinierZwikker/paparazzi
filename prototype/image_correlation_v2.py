from PIL import Image
from os import walk, mkdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


selected_images = range(105, 140)  # 'all' or specific [###, ###, ..., ###] or range(start, stop, step)
amount_of_steps = 25
kernel_size = 50

kernel_extend = int(kernel_size / 2)

# == Loading dataset ==
file_names = []
for _, _, file_names in walk("../data/dataset/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935"):
    break

images = []
for file_name in file_names:
    with Image.open(f"../data/dataset/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/{file_name}") as image_file:
        images.append({'img': np.array(image_file.rotate(90, expand=True)), 'time': float(file_name.removesuffix('.jpg'))/(10**6), 'name': file_name})

images = sorted(images, key=lambda x: x['time'])

with open("../data/dataset/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142943.csv") as csv_file:
    dataframe = pd.read_csv(csv_file)

locations = []      # Location of Eval points in [x,y]
directions = []     # (Positive) Eval direction in [dx,dy]
dependencies = []   # 0 for radial (forward motion), 1 for sideways motion (+ towards right)

radial_lists = [[[-30, -15, 0, 15, 30, 150, 165, 180, -165, -150], [30, 60, 90, 120, 140, 160, 180, 210]],
                [[-22, -7, 7, 22, 158, 173, -173, -158], [120, 140, 160, 180, 210]],
                [[65, 90, 115, -45, -60, -75, -90, -105, -120, -135], [40, 60, 80]]]

sideways_lines = [0, -50, 50]
sideways_resolution = 40

manual_sideways = [[0, 25], [20, 25], [40, 25], [-20, 25], [-40, 25],
                   [0, -25], [20, -25], [40, -25], [-20, -25], [-40, -25]]

manual_incoming = [[0, -95], [20, -95], [40, -95], [-20, -95], [-40, -95],
                   [0, -75], [20, -75], [40, -75], [-20, -75], [-40, -75],
                   [0, -55], [20, -55], [40, -55], [-20, -55], [-40, -55]]

for radial_list in radial_lists:
    for radial in radial_list[0]:
        for distance in radial_list[1]:
            radial_rad = radial / 180 * np.pi
            position_x = 260 + distance * np.cos(radial_rad)
            position_y = 120 + distance * np.sin(radial_rad)
            if kernel_extend < position_x < 520 - kernel_extend:
                if kernel_extend < position_y < 240 - kernel_extend:
                    locations.append([position_x, position_y])
                    directions.append([np.cos(radial_rad), np.sin(radial_rad)])
                    dependencies.append(0)

for sideways_line in sideways_lines:
    for distance in range(0, 260, sideways_resolution):
        position_x = 260 + distance
        position_y = 120 + sideways_line
        if kernel_extend < position_x < 520 - kernel_extend:
            if kernel_extend < position_y < 240 - kernel_extend:
                locations.append([position_x, position_y])
                directions.append([1, 0])
                dependencies.append(1)
        position_x = 260 - distance
        if kernel_extend < position_x < 520 - kernel_extend:
            if kernel_extend < position_y < 240 - kernel_extend:
                locations.append([position_x, position_y])
                directions.append([1, 0])
                dependencies.append(1)

for position in manual_sideways:
    locations.append([260 + position[0], 120 + position[1]])
    directions.append([1, 0])
    dependencies.append(1)

for position in manual_incoming:
    locations.append([260 + position[0], 120 + position[1]])
    directions.append([0, -1])
    dependencies.append(0)

locations_x = np.array([location[0] for location in locations]).astype(np.uint16)
locations_y = np.array([location[1] for location in locations]).astype(np.uint16)
directions_x = np.array([direction[0] for direction in directions]).astype(np.float64)
directions_y = np.array([direction[1] for direction in directions]).astype(np.float64)
dependencies = np.array(dependencies).astype(np.uint8)

if (len(locations_y) != len(locations_x) or
    len(locations_y) != len(directions_x) or
    len(locations_y) != len(directions_y) or
    len(locations_y) != len(dependencies)):
    raise IndexError(f"Not all arrays have the same length! {len(locations_y)}/{len(locations_x)}/{len(directions_x)}/{len(directions_x)}/{len(dependencies)}")

amount_of_locations = len(locations_y)

print(f"x: {locations_x}\n\ny: {locations_y}\n\ndir x: {directions_x}\n\ndir_y: {directions_y}\n\ndep: {dependencies}\n\nAmount of evaluation locations: {amount_of_locations}")

for current_image_i in [195]:  # range(len(images)):
    previous_image = images[current_image_i - 1]['img']
    current_image = images[current_image_i]['img']

    fig = plt.figure(figsize=(8, 4), layout='constrained')
    # plt.imshow(images[current_image]['img'], alpha=0.5)
    plt.imshow(current_image, alpha=0.5)
    plt.plot(locations_x[dependencies == 0], locations_y[dependencies == 0], 'rx')
    plt.plot(locations_x[dependencies == 0], locations_y[dependencies == 0], 'rs',
             linestyle='none', markerfacecolor='none', markersize=kernel_size)
    plt.plot(locations_x[dependencies == 1], locations_y[dependencies == 1], 'bx')
    plt.plot(locations_x[dependencies == 1], locations_y[dependencies == 1], 'bs',
             linestyle='none', markerfacecolor='none', markersize=kernel_size)
    plt.show()

    results = np.zeros((amount_of_locations, amount_of_steps))

    # == Find corresponding data line ==
    time_index = dataframe['time'].sub(images[current_image_i]['time']).abs().idxmin()

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

    print(f"\tWorld Velocity:  [{vel_world[0]: 1.2f}, {vel_world[1]: 1.2f}, {vel_world[2]: 1.2f}] (x,y,z) [m/s]\n"
          f"\tBody velocity:   [{vel_body[0]: 1.2f}, {vel_body[1]: 1.2f}, {vel_body[2]: 1.2f}] (x,y,z) [m/s]\n")


    for eval_i in range(amount_of_locations):

        previous_slice = previous_image[
                             int(locations_y[eval_i] - kernel_extend):int(locations_y[eval_i] + kernel_extend),
                             int(locations_x[eval_i] - kernel_extend):int(locations_x[eval_i] + kernel_extend),
                             :]

        previous_slice = previous_slice / np.linalg.norm(previous_slice)

        for step_i in range(amount_of_steps):
            if dependencies[eval_i] == 0:
                speed_factor = vel_body[1]
            else:
                speed_factor = vel_body[0]
            if 0 < speed_factor < 1:
                speed_factor = 1
            if 0 > speed_factor > -1:
                speed_factor = -1

            current_location_x = locations_x[eval_i] + step_i * directions_x[eval_i] * speed_factor
            current_location_y = locations_y[eval_i] + step_i * directions_y[eval_i] * speed_factor

            if 0 < current_location_x - kernel_extend < 520 and \
               0 < current_location_x + kernel_extend < 520 and \
               0 < current_location_y - kernel_extend < 240 and \
               0 < current_location_y + kernel_extend < 240:

                current_slice = current_image[
                                 int(current_location_y - kernel_extend):int(current_location_y + kernel_extend),
                                 int(current_location_x - kernel_extend):int(current_location_x + kernel_extend),
                                 :]

                current_slice = current_slice / np.linalg.norm(current_slice)

                results[eval_i, step_i] = np.sum(np.multiply(previous_slice,
                                                             current_slice))

                # print(eval_i, step_i, results[eval_i, step_i])
            else:
                results[eval_i, step_i] = np.nan


    stds = []
    max_indices = np.zeros(amount_of_locations)

    fig = plt.figure(figsize=(8, 4), layout='constrained')
    for row_i in range(amount_of_locations):
        stds.append(np.std(results[row_i, :]))
        if np.std(results[row_i, :]) < 0.02:
            results[row_i, :] = np.nan
        else:
            plt.plot(results[row_i, :], 'k')
        max_indices[row_i] = np.argmax(results[row_i, :]) / amount_of_steps
        if dependencies[row_i] == 0:
            max_indices[row_i] /= vel_body[1]
        else:
            max_indices[row_i] /= vel_body[0]
    plt.show()

    max_indices /= max(max(max_indices), 1)

    fig = plt.figure(figsize=(8, 2), layout='constrained')
    plt.imshow(results.transpose())
    plt.show()

    fig = plt.figure(figsize=(8, 4), layout='constrained')
    plt.plot(stds, 'k')
    plt.show()

    fig = plt.figure(figsize=(8, 4), layout='constrained')
    plt.plot(max_indices, 'k')
    plt.show()

    placed_result = np.zeros_like(current_image)
    for eval_i in range(amount_of_locations):
        placed_result[locations_y[eval_i], locations_x[eval_i]] = 255 * np.array((max_indices[eval_i],
                                                                                  max_indices[eval_i],
                                                                                  max_indices[eval_i]))


    # plt.imsave(f"../data/renders/v2/original/{current_image_i}.png", current_image)
    # plt.imsave(f"../data/renders/v2/depth/{current_image_i}.png", placed_result)

    fig = plt.figure(figsize=(8, 4), layout='constrained')
    plt.imshow(current_image, alpha=0.5)
    plt.imshow(placed_result, alpha=0.5)
    plt.show()



