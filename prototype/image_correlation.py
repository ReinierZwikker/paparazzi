from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import walk
import pandas as pd

file_names = []
for _, _, file_names in walk("../data/dataset/AE4317_2019_datasets/cyberzoo_poles/20190121-135009"):
    # print(file_names)
    break


images = []
for file_name in file_names:
    with Image.open(f"../data/dataset/AE4317_2019_datasets/cyberzoo_poles/20190121-135009/{file_name}") as image_file:
        images.append({'img': np.array(image_file.rotate(90, expand=True)), 'time': float(file_name.removesuffix('.jpg'))/(10**6)})

images = sorted(images, key=lambda x: x['time'])

image_size = images[0]['img'].shape

with open("../data/dataset/AE4317_2019_datasets/cyberzoo_poles/20190121-135121.csv") as csv_file:
    dataframe = pd.read_csv(csv_file)

print(dataframe)

current_image = 297

print(images[current_image]['time'])

time_index = dataframe['time'].sub(images[current_image]['time']).abs().idxmin()

print(time_index)

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

rectangle_size = (20, 20, 1)
rectangle_extend = (int(rectangle_size[0]/2), int(rectangle_size[1]/2), rectangle_size[2])
amount_of_rectangles = np.prod(image_size)
print(f"\nAmount of rectangles for total convolution: {amount_of_rectangles}")


def correlate_line(image, kernel_source, kernel_size, start_x, start_y, end_x, end_y, step=1, logarithmic_spacing=True, plot=True):
    """

    :param image:
    :param kernel_source:
    :param kernel_size:
    :param start_x: TODO make into (x, y)
    :param start_y:
    :param end_x: TODO make into (x, y)
    :param end_y:
    :param step:
    :param logarithmic_spacing:
    :param plot:
    :return:
    """
    # TODO Allow for different image and kernel lines
    amount_of_line_points = int(((start_x - end_x)**2 + (start_y - end_y)**2)**0.5 / step)
    print(f"\n=== LINE CONVOLUTION ===\n"
          f"\tfrom ({start_x}, {start_y}) to ({end_x}, {end_y}) in {amount_of_line_points} steps")

    if logarithmic_spacing:
        x_positions = end_x - np.floor(np.geomspace(1, end_x - start_x, amount_of_line_points)).astype(int)
        y_positions = end_y - np.floor(np.geomspace(1, end_y - start_y, amount_of_line_points)).astype(int)
    else:
        x_positions = np.floor(np.linspace(start_x, end_x, amount_of_line_points)).astype(int)
        y_positions = np.floor(np.linspace(start_y, end_y, amount_of_line_points)).astype(int)

    results = np.zeros((amount_of_line_points, amount_of_line_points))
    kernel_extend = (int(np.floor((kernel_size[0] / 2))), int(np.floor((kernel_size[1] / 2))))
    print(f"\tKernel Extend: {kernel_extend}")
    for i in range(amount_of_line_points):
        kernel = kernel_source[
                         int(x_positions[i] - kernel_extend[0]):int(x_positions[i] + kernel_extend[0]),
                         int(y_positions[i] - kernel_extend[1]):int(y_positions[i] + kernel_extend[1]),
                         :]
        kernel = kernel / np.linalg.norm(kernel)
        for j in range(amount_of_line_points):
            # if abs(i - j) < 20:
            image_slice = image[
                          int(x_positions[j] - kernel_extend[0]):int(x_positions[j] + kernel_extend[0]),
                          int(y_positions[j] - kernel_extend[1]):int(y_positions[j] + kernel_extend[1]),
                          :]
            image_slice = image_slice / np.linalg.norm(image_slice)
            results[i, j] = np.sum(np.multiply(image_slice,
                                               kernel))
            # results[i, i-j]
            # print(f"({x_positions[i]}, {y_positions[i]}): {results[i, j]}")
    if plot:
        plt.plot(y_positions, x_positions, 'rs', linestyle='none', markerfacecolor='none', markersize=kernel_size[0])

    return results, x_positions, y_positions


def sweep_around(center_point, image, kernel_source, kernel_size=(10, 10),
                 sweep_resolution=5, line_resolution=10,
                 rotation=0, kernel_center_point=None):
    """
    TODO

    :param center_point:
    :param image:
    :param kernel_source:
    :param kernel_size:
    :param sweep_resolution: resolution in pixels on the edge of the image
    :param line_resolution: resolution in pixels of the box offset on the line
    :param rotation:
    :param kernel_center_point:
    """
    if kernel_center_point is None:
        kernel_center_point = center_point
    else:
        raise NotImplementedError("Shift of kernel center point not yet implemented")
    if rotation != 0:
        raise NotImplementedError("Rotation in image not yet implemented")
    # TODO implement rotation and kernel center point
    kernel_extend = (int(np.floor((kernel_size[0] / 2))), int(np.floor((kernel_size[1] / 2))))

    # Brute force sweep
    # TODO make neater and divide more equally over image
    end_points = []
    i = 5 * kernel_extend[0]
    j = 5 * kernel_extend[1]
    while i < image.shape[0] - 5 * kernel_extend[0]:
        end_points.append((i, j))
        i += sweep_resolution
    while j < image.shape[1] - 5 * kernel_extend[1]:
        end_points.append((i, j))
        j += sweep_resolution
    while i > 5 * kernel_extend[0]:
        end_points.append((i, j))
        i -= sweep_resolution
    while j > 5 * kernel_extend[1]:
        end_points.append((i, j))
        j -= sweep_resolution
    print(len(end_points))

    results = []

    for end_point in end_points:
        results.append(correlate_line(image, kernel_source, kernel_size,
                                      center_point[0], center_point[1], end_point[0], end_point[1],
                                      step=line_resolution, logarithmic_spacing=False))

    return results


fig = plt.figure(figsize=(8, 4), layout='constrained')
plt.title(str(images[10]['time']) + " + " + str(images[11]['time']))
plt.imshow(images[current_image]['img'], alpha=0.5)
plt.imshow(images[current_image + 1]['img'], alpha=0.5)

# TEST CORRELATE LINE
line_start = (15, 15)
line_end = (125, 300)
kernel_size = (4, 4)

line_result = correlate_line(images[current_image + 1]['img'], images[current_image]['img'], kernel_size,
                             line_start[0], line_start[1], line_end[0], line_end[1], step=2, logarithmic_spacing=False)
plt.show()

fig = plt.figure(figsize=(6, 6), layout='constrained')
plt.title(f"Line Correlation result from {line_start} to {line_end}")
plt.imshow(line_result[0])
plt.show()

# TEST CORRELATE SWEEP
test_center_point = (100, 100)
test_kernel_size = (4, 4)


fig = plt.figure(figsize=(8, 4), layout='constrained')
plt.title(str(images[10]['time']) + " + " + str(images[11]['time']))
plt.imshow(images[current_image]['img'], alpha=0.5)
plt.imshow(images[current_image + 1]['img'], alpha=0.5)

sweep_result = sweep_around(test_center_point, images[current_image + 1]['img'], images[current_image]['img'],
                            test_kernel_size, sweep_resolution=20, line_resolution=2)

plt.show()

print(len(sweep_result))
for line in [0, 10, 20, 30, 40, 50, 60]:
    fig = plt.figure(figsize=(6, 6), layout='constrained')
    plt.title(f"Line Correlation result of Line {line}")
    plt.imshow(sweep_result[line][0])
    plt.show()

# # Correlation EXAMPLE
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