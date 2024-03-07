import numpy as np
import matplotlib.pyplot as plt


def bound(value, lower, upper):
    return max(lower, min(upper, value))


def correlate_line(image, kernel_source, kernel_size,
                   start_x, start_y, end_x, end_y, step=1,
                   logarithmic_spacing=True, plot=False):
    """

    :param image:
    :param kernel_source:
    :param kernel_size:
    :param start_x: TODO make into (x, y)
    :param start_y:
    :param end_x: TODO make into (x, y)
    :param end_y:
    :param step:
    :param search_distance:
    :param logarithmic_spacing:
    :param plot:
    :return:
    """
    # TODO Allow for different image and kernel lines
    amount_of_line_points = int(((start_x - end_x)**2 + (start_y - end_y)**2)**0.5 / step)
    print(f"\t== LINE CONVOLUTION ==\n"
          f"\t\tfrom ({start_x}, {start_y}) to ({end_x}, {end_y}) in {amount_of_line_points} steps")

    if logarithmic_spacing:
        x_positions = end_x - np.floor(np.geomspace(1, end_x - start_x, amount_of_line_points)).astype(int)
        y_positions = end_y - np.floor(np.geomspace(1, end_y - start_y, amount_of_line_points)).astype(int)
    else:
        x_positions = np.floor(np.linspace(start_x, end_x, amount_of_line_points)).astype(int)
        y_positions = np.floor(np.linspace(start_y, end_y, amount_of_line_points)).astype(int)

    results = np.zeros((amount_of_line_points, amount_of_line_points))
    kernel_extend = (int(np.floor((kernel_size[0] / 2))), int(np.floor((kernel_size[1] / 2))))
    for i in range(amount_of_line_points):
        kernel = kernel_source[
                         int(x_positions[i] - kernel_extend[0]):int(x_positions[i] + kernel_extend[0]),
                         int(y_positions[i] - kernel_extend[1]):int(y_positions[i] + kernel_extend[1]),
                         :]
        kernel = kernel / np.linalg.norm(kernel)
        for j in range(amount_of_line_points):
            if abs(i - j) < search_distance / 2:
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
                 sweep_resolution=5, line_resolution=10, search_distance=20,
                 rotation=0, kernel_center_point=None,
                 plot=False):
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

    # ======
    # Brute force sweep
    # TODO make neater and divide more equally over image
    # 5* offset temporary to make sure we never overrun
    # the boundaries of the image, not for final algorithm!
    # ======
    # end_points = []
    # i = 5 * kernel_extend[0]
    # j = 5 * kernel_extend[1]
    # while i < image.shape[0] - 5 * kernel_extend[0]:
    #     end_points.append((i, j))
    #     i += sweep_resolution
    # while j < image.shape[1] - 5 * kernel_extend[1]:
    #     end_points.append((i, j))
    #     j += sweep_resolution
    # while i > 5 * kernel_extend[0]:
    #     end_points.append((i, j))
    #     i -= sweep_resolution
    # while j > 5 * kernel_extend[1]:
    #     end_points.append((i, j))
    #     j -= sweep_resolution

    # Circular Sweep
    end_points = []
    for angle in range(0, 360, sweep_resolution):
        angle_rad = angle / 180 * np.pi
        end_points.append([int(120+110*np.cos(angle_rad)),
                           int(260+250*np.sin(angle_rad))])
    results = []

    for end_point in end_points:
        results.append(correlate_line(image, kernel_source, kernel_size,
                                      center_point[0], center_point[1], end_point[0], end_point[1],
                                      step=line_resolution, search_distance=search_distance,
                                      logarithmic_spacing=False, plot=plot))

    return results


def sweep_horizontal(image, kernel_source, inter_line_distance, line_resolution, search_distance, kernel_size=(10, 10), plot=False):
    start_x = 10
    end_x = 510

    y_locations = np.arange(10, 240, inter_line_distance)
    results = []

    for y in y_locations:
        results.append(correlate_line(image, kernel_source, kernel_size,
                                      y, start_x, y, end_x,
                                      step=line_resolution, search_distance=search_distance,
                                      logarithmic_spacing=False, plot=plot))
    return results


def find_depth(radial_sweep_result, depth_map, confidence_map):
    """

    :param radial_sweep_result: result of one radial sweep, as (result, x_positions, y_positions)
    :param depth_map: 2D depth map of image, in meters
    :param confidence_map: 2D confidence map of the depth map
    :return: modified depth_image
    """
    for i in range(radial_sweep_result[0].shape[0]):
        max_index = np.argmax(radial_sweep_result[0][i, :])
        depth_map[radial_sweep_result[1][i], radial_sweep_result[2][i]] = int(2550 * abs(i - max_index) / radial_sweep_result[0].shape[0])
        confidence_map[radial_sweep_result[1][i], radial_sweep_result[2][i]] = 2550 * (radial_sweep_result[0][i, max_index] - np.mean(radial_sweep_result[0][i, :]))
    return depth_map, confidence_map

def convert_depth_to_td_map(depth_map, factor=1):
    depth_map = np.mean(depth_map.astype(np.uint8), axis=2)
    td_map = np.zeros_like(depth_map)
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth = (depth_map[i, j] / 255 * 239 * factor).astype(np.uint8)
            td_map[depth, j] = 255
    td_map = np.expand_dims(td_map, axis=2)
    td_map = np.repeat(td_map, 3, axis=2)

    return td_map
