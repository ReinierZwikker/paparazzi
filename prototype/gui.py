import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget


class MainWindow(QMainWindow):

    def __init__(self, base_path: str = "../data/renders"):
        self.base_path: str = base_path
        self.date: str = "20240305_152134"
        self.image_paths: list = self.load_directory()
        self.selected_image: str = "Select Image"
        self.image_list: list = ["original", "mixed", "squares", "depth", "confidence", "blur", "memory"]  # ["original", "mixed", "squares", "depth", "confidence", "combined", "blur"]

        # Init window
        super(MainWindow, self).__init__()
        self.title = "Image Viewer"
        self.setWindowTitle(self.title)
        self.central_label = QLabel(self)
        self.outer_layout = QVBoxLayout(self.central_label)

        # Buttons
        self.button_widget = QWidget()
        self.button_widget.setFixedSize(600, 50)
        self.button_layout = QHBoxLayout(self.button_widget)

        # File selection
        self.file_browser_btn = QPushButton(self)
        self.file_browser_btn.setText(self.selected_image)
        self.file_browser_btn.clicked.connect(self.open_file_dialog)
        self.button_layout.addWidget(self.file_browser_btn)

        button_run = QPushButton(self)
        button_run.setText("Run")
        button_run.clicked.connect(self.button_run_clicked)
        self.button_layout.addWidget(button_run)

        button_previous = QPushButton(self)
        button_previous.setText("Previous")
        button_previous.clicked.connect(self.button_previous_clicked)
        self.button_layout.addWidget(button_previous)

        button_next = QPushButton(self)
        button_next.setText("Next")
        button_next.clicked.connect(self.button_next_clicked)
        self.button_layout.addWidget(button_next)

        # Pictures
        self.picture_widget_1 = QWidget()
        self.picture_widget_2 = QWidget()
        self.picture_layout_1 = QHBoxLayout(self.picture_widget_1)
        self.picture_layout_2 = QHBoxLayout(self.picture_widget_2)
        for i, image_folder in enumerate([self.image_list]):
            label = QLabel()
            pixmap = QPixmap(f'{self.base_path}/{self.date}/{image_folder}/{self.selected_image}.png')
            label.setPixmap(pixmap)
            if i < 4:
                self.picture_layout_1.addWidget(label)
            else:
                self.picture_layout_2.addWidget(label)

        self.picture_widget = QWidget()
        self.picture_layout = QVBoxLayout(self.picture_widget)
        self.picture_layout.addWidget(self.picture_widget_1)
        self.picture_layout.addWidget(self.picture_widget_2)

        # Plot the window
        self.outer_layout.addWidget(self.button_widget)
        self.outer_layout.addWidget(self.picture_widget)
        self.setCentralWidget(self.central_label)
        self.resize(650, 300)

    def load_directory(self):
        files = os.listdir(f"{self.base_path}/{self.date}/original")
        return files

    def open_file_dialog(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(f'C:/Master/Operation Optimization/paparazzi/data')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                if len(filenames) == 1:
                    if filenames[0][-4:] == ".jpg":
                        image_name_path = filenames[0].removesuffix('.jpg')
                        # self.date = image_name_path.split("/")[-2]
                        self.selected_image = image_name_path.split("/")[-1]

        self.re_init()

    def re_init(self):
        # Init window
        self.central_label = QLabel(self)
        self.outer_layout = QVBoxLayout(self.central_label)

        # Buttons
        self.button_widget = QWidget()
        self.button_widget.setFixedSize(600, 50)
        self.button_layout = QHBoxLayout(self.button_widget)

        # File selection
        self.file_browser_btn = QPushButton(self)
        self.file_browser_btn.setText(self.selected_image)
        self.file_browser_btn.clicked.connect(self.open_file_dialog)
        self.button_layout.addWidget(self.file_browser_btn)

        button_run = QPushButton(self)
        button_run.setText("Run")
        button_run.clicked.connect(self.button_run_clicked)
        self.button_layout.addWidget(button_run)

        button_previous = QPushButton(self)
        button_previous.setText("Previous")
        button_previous.clicked.connect(self.button_previous_clicked)
        self.button_layout.addWidget(button_previous)

        button_next = QPushButton(self)
        button_next.setText("Next")
        button_next.clicked.connect(self.button_next_clicked)
        self.button_layout.addWidget(button_next)

        # Pictures
        self.picture_widget_1 = QWidget()
        self.picture_widget_2 = QWidget()
        self.picture_layout_1 = QHBoxLayout(self.picture_widget_1)
        self.picture_layout_2 = QHBoxLayout(self.picture_widget_2)

        self.change_image()

        self.picture_widget = QWidget()
        self.picture_layout = QVBoxLayout(self.picture_widget)
        self.picture_layout.addWidget(self.picture_widget_1)
        self.picture_layout.addWidget(self.picture_widget_2)

        # Plot the window
        self.outer_layout.addWidget(self.button_widget)
        self.outer_layout.addWidget(self.picture_widget)
        self.setCentralWidget(self.central_label)
        # self.resize(2300, 1200)


    def change_image(self):
        for i, image_folder in enumerate(self.image_list):
            label = QLabel()
            pixmap = QPixmap(f'{self.base_path}/{self.date}/{image_folder}/{self.selected_image}.png')
            label.setPixmap(pixmap)
            if i < 4:
                self.picture_layout_1.addWidget(label)
            else:
                self.picture_layout_2.addWidget(label)

        self.file_browser_btn.setText(self.selected_image)

    def button_next_clicked(self):
        if self.selected_image != "Select Image":
            new_index = self.image_paths.index(f"{self.selected_image}.png") + 1
            if new_index < len(self.image_paths):
                self.selected_image = self.image_paths[new_index].removesuffix('.png')
        self.re_init()

    def button_previous_clicked(self):
        if self.selected_image != "Select Image":
            new_index = self.image_paths.index(f"{self.selected_image}.png") - 1
            if new_index < len(self.image_paths):
                self.selected_image = self.image_paths[new_index].removesuffix('.png')
        self.re_init()

    def button_run_clicked(self):
        print("run something")


app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())
