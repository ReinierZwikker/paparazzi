import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QFileDialog


with open("..\\data\\mavlab_tests\\20240301-095957.csv") as csv_file:
    dataframe = pd.read_csv(csv_file)


class MainWindow(QMainWindow):

    def __init__(self, base_path: str = "../data/mavlab_tests/01_03_2024"):
        self.base_path: str = base_path
        self.image_paths: list = self.load_directory()
        self.current_image_id: int = 0

        self.play = False

        super(MainWindow, self).__init__()
        self.title = "Image Viewer"
        self.setWindowTitle(self.title)

        self.central_label = QLabel(self)
        self.label_attitude = QLabel('Attitude: 0, 0, 0', self)
        self.label_rates = QLabel('Position: 0, 0, 0', self)
        self.label_rates.move(0, 15)

        # Buttons
        button_next = QPushButton(self)
        button_next.setText("Next")
        button_next.move(420, 270)
        button_next.clicked.connect(self.button_next_clicked)

        button_previous = QPushButton(self)
        button_previous.setText("Previous")
        button_previous.move(320, 270)
        button_previous.clicked.connect(self.button_previous_clicked)

        # File selection
        self.file_browser_btn = QPushButton(self)
        self.file_browser_btn.setText(self.image_paths[self.current_image_id])
        self.file_browser_btn.move(0, 270)
        self.file_browser_btn.clicked.connect(self.open_file_dialog)

        # Picture
        pixmap = QPixmap(f'{self.base_path}/{self.image_paths[self.current_image_id]}')
        pixmap = pixmap.transformed(QTransform().rotate(-90))

        # Plot the window
        self.central_label.setPixmap(pixmap)
        self.setCentralWidget(self.central_label)
        self.resize(520, 300)

    def load_directory(self):
        files = os.listdir(self.base_path)
        return files

    def get_log_data(self):
        time_stamp = float(self.image_paths[self.current_image_id].removesuffix('.jpg'))/(10**6)
        time_index = dataframe['time'].sub(time_stamp).abs().idxmin()
        print(dataframe.iloc[time_index])
        attitude = (np.rad2deg(dataframe.iloc[time_index]['att_phi']),
                    np.rad2deg(dataframe.iloc[time_index]['att_theta']),
                    np.rad2deg(dataframe.iloc[time_index]['att_psi']))

        # (x_world, y_world, z_world)
        rates = (np.rad2deg(dataframe.iloc[time_index]['rate_p']),
                 np.rad2deg(dataframe.iloc[time_index]['rate_q']),
                 np.rad2deg(dataframe.iloc[time_index]['rate_r']))

        self.label_attitude.setText(f'Attitude: {attitude[0]}, {attitude[1]}, {attitude[2]}')
        self.label_attitude.adjustSize()
        self.label_rates.setText(f'Rates: {rates[0]}, {rates[1]}, {rates[2]}')
        self.label_rates.adjustSize()

    def open_file_dialog(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(f'C:/Master/Operation Optimization/paparazzi/data/mavlab_tests/01_03_2024')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                image_name = filenames[0].split("/")[-1]
                self.current_image_id = self.image_paths.index(image_name)
        self.change_image()

    def change_image(self):
        pixmap = QPixmap(f'{self.base_path}/{self.image_paths[self.current_image_id]}')
        pixmap = pixmap.transformed(QTransform().rotate(-90))
        self.get_log_data()
        self.central_label.setPixmap(pixmap)
        self.file_browser_btn.setText(self.image_paths[self.current_image_id])


    def button_next_clicked(self):
        self.current_image_id += 1
        self.change_image()

    def button_previous_clicked(self):
        self.current_image_id -= 1
        self.change_image()


app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())
