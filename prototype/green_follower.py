import numpy as np
from utils import Camera
import matplotlib.pyplot as plt


camera = Camera("cyberzoo_poles_panels_mats", "20190121-142935")
camera.rotate(90)
camera.select_frame("24182668.jpg")
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
        radial_array[i] = image[row, column, 1]

    return np.sum(radial_array)


def get_direction(resolution: int, image: np.array):
    radials: np.array = np.empty(resolution)
    angles: np.array = np.linspace(0, np.pi, resolution)
    for i, angle in enumerate(angles):
        radials[i] = get_radial(angle, image, radius=int(image.shape[0]/2))

    return 90 - np.rad2deg(angles[np.argmax(radials)])


if __name__ == "__main__":
    print(get_direction(50, camera.image_array))

    fig, ax = plt.subplots()
    ax.imshow(camera.image_array[:, :, 1])
    plt.show()
