import numpy as np
from utils import Camera
import matplotlib.pyplot as plt


camera = Camera("own_data", "20240301-095957")
camera.rotate(90)
camera.select_frame("311847611.jpg")
print(camera)


def get_radial(angle: float, image: np.array, radius: int = 120):
    if 0 <= angle <= np.pi:
        centre = int(image.shape[1]/2)
    else:
        raise Exception("Angle out of bounds")

    bottom = image.shape[0] - 1

    radial_array: np.array = np.empty(radius)
    for i in range(radius):
        row = bottom - int(i * np.sin(angle))
        column = centre + int(i * np.cos(angle))
        radial_array[i] = np.sum(image[row, column])

    return np.sum(radial_array) * (np.sin(angle) + 0.2)


def get_direction(resolution: int, image: np.array):
    radials: np.array = np.empty(resolution)
    angles: np.array = np.linspace(0, np.pi, resolution)
    for i, angle in enumerate(angles):
        radials[i] = get_radial(angle, image, radius=int(image.shape[0]/2))

    return angles[np.argmax(radials)]


def create_threshold_image(image: np.array):
    threshold_yuv_image = np.zeros_like(camera.image_yuv, dtype=int)
    threshold_yuv_image[:, :, 0][np.logical_and(camera.image_yuv[:, :, 0] > 60, camera.image_yuv[:, :, 0] < 130)] = 255
    threshold_yuv_image[:, :, 1][np.logical_and(camera.image_yuv[:, :, 0] > 75, camera.image_yuv[:, :, 0] < 110)] = 255
    threshold_yuv_image[:, :, 2][np.logical_and(camera.image_yuv[:, :, 0] > 120, camera.image_yuv[:, :, 0] < 140)] = 255
    return threshold_yuv_image


if __name__ == "__main__":

    threshold_image = create_threshold_image(camera.image_yuv)

    direction_angle = get_direction(50, threshold_image)
    print(f"Direction: {round(90 - np.rad2deg(direction_angle),2)} degrees")

    fig, ax = plt.subplots()
    ax.imshow(threshold_image.astype(int))
    ax.plot([260, 260 + 240 * np.cos(direction_angle)], [240, 240 - 240 * np.sin(direction_angle)], color="#00FF00")
    ax.set_xlim(0, 520)
    ax.set_ylim(240, 0)
    plt.show()
