import os
import numpy as np
from PIL import Image


class Camera:
    def __init__(self, dataset: str, date: str, base_path: str = "../data/datasets"):
        """
        Creates a way to easily browse through a dataset and convert it to the right datatypes.
        Make sure you put the datasets within the data folder.

        param dataset: The name of the folder within datasets which you want to view
        param date: The name of the folder within the dataset which you want to view

        Attributes are made private to prevent conversion mismatches, access them through @property defined below
        """
        self._base_path: str = base_path
        self._dataset: str = dataset
        self._date: str = date

        self._rotation: float = 0

        self._images_in_directory: list[str] = self.load_current_directory()
        self._selected_image: int = 0
        self._image_path: str = os.path.join(self._base_path, self._dataset, self._date, self._images_in_directory[self._selected_image])

        self._current_image: Image = self.load_selected_image()
        self._current_image_array: np.array = self.convert_image_to_array()  # [rows, columns, rgb]
        self._current_image_yuv: np.array = self.convert_image_to_yuv()  # [rows, columns, yuv]
        self._current_time: int = int(self._images_in_directory[self._selected_image].removesuffix('.jpg'))  # microseconds

    def update_image(self):
        self._image_path: str = os.path.join(self._base_path, self._dataset, self._date, self._images_in_directory[self._selected_image])
        self._current_image: Image = self.load_selected_image()
        self._current_image_array: np.array = self.convert_image_to_array()  # [rows, columns, rgb]
        self._current_time: int = int(self._images_in_directory[self._selected_image].removesuffix('.jpg'))  # microseconds

    def load_current_directory(self):
        return os.listdir(os.path.join(self._base_path, self._dataset, self._date))

    def load_selected_image(self):
        image = Image.open(self._image_path)
        return image.rotate(self._rotation, Image.NEAREST, expand=True)

    def convert_image_to_array(self):
        return np.array(self._current_image)

    def convert_image_to_yuv(self):
        m = np.array([[0.29900, -0.16874, 0.50000],
                      [0.58700, -0.33126, -0.41869],
                      [0.11400, 0.50000, -0.08131]])

        yuv = np.dot(self._current_image_array, m)
        yuv[:, :, 1:] += 128.0
        return yuv

    def next_frame(self):
        self._selected_image += 1
        self.update_image()

    def previous_frame(self):
        self._selected_image -= 1
        self.update_image()

    def select_frame(self, image_name):
        self._selected_image = self._images_in_directory.index(image_name)
        self.update_image()

    def rotate(self, degrees: float):
        self._rotation = degrees
        self.update_image()

    @property
    def image(self):
        return self._current_image

    @property
    def image_rgb(self):
        return self._current_image_array

    @property
    def image_yuv(self):
        return self._current_image_yuv

    @property
    def time(self):
        return self._current_time

    @property
    def image_path(self):
        return self._image_path

    def __str__(self):
        image_path = self._images_in_directory[self._selected_image]
        return f"Camera Image:\n" \
               f"\tDirectory:  {self._base_path}/{self._dataset}/{self._date}/{image_path}\n" \
               f"\tSize:       {self._current_image_array.shape}\n" \
               f"\tTime Stamp: {round(self._current_time * 10**(-6), 6)} seconds"


if __name__ == "__main__":
    camera = Camera("own_data", "20240301-095957")
    print(camera)
