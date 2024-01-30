import os, sys, math, pdb
import re
import cv2
import numpy as np
import scipy.signal as sig
import scipy.ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot,Qt, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.QtGui import QIcon,QImage,QPixmap, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QMainWindow,QMenu,QMenuBar,QWidget,QAction, QApplication, QFileDialog, QMessageBox, QListView, QCompleter, QLineEdit,QListWidget, QListWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from project import Ui_MainWindow

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)

        self.setWindowTitle("First Semester Project")
        app.setStyle("Fusion")

        self.data = pd.read_csv('C:/Users/Etudiant FST/Desktop/Project/updated_file.csv')
        # Help and Information
        self.ui.label_3.setText('<a style="color:black;" href="https://www.google.com">Help & Information</a>')
        self.ui.label_3.setOpenExternalLinks(True)
        self.ui.label_3.linkActivated.connect(self.open_url)

############## Image Attributes############
        self.ui.image = None
        self.tmp = None
        self.ui.actionOpen.triggered.connect(self.open_img)
        self.ui.actionSave.triggered.connect(self.save_img)
        self.ui.actionPrint.triggered.connect(self.createPrintDialog)
        self.ui.actionQuit.triggered.connect(self.QuestionMessage)
        self.ui.actionBig.triggered.connect(self.big_Img)
        self.ui.actionSmall.triggered.connect(self.small_Img)

        # Set input
        self.ui.dial.valueChanged.connect(self.rotation)
        self.ui.horizontalSlider.valueChanged.connect(self.Gamma_)
        self.ui.gaussian_QSlider.valueChanged.connect(self.gaussian_filter2)
        self.ui.erosion.valueChanged.connect(self.erode)
        self.ui.Qlog.valueChanged.connect(self.Log)
        # self.ui.size_Img.valueChanged.connect(self.SIZE)
        self.ui.canny.stateChanged.connect(self.Canny)
        self.ui.canny_min.valueChanged.connect(self.Canny)
        self.ui.canny_max.valueChanged.connect(self.Canny)
        self.ui.pushButton.clicked.connect(self.reset)

        # Transformations
        self.ui.actionAffine.triggered.connect(self.shearing)
        self.ui.actionTranslation.triggered.connect(self.translation)

        # Simple Edge Detection
        self.ui.actionSHT.triggered.connect(self.hough_transform_edge_detection)
        self.ui.actionDirectional_Filtering_2.triggered.connect(self.directional_filtering)

        # Filter
        self.ui.actionBlur.triggered.connect(self.blur)
        self.ui.actionBox_Filter.triggered.connect(self.box_filter)
        self.ui.actionMedian_Filter.triggered.connect(self.median_filter)
        self.ui.actionBilateral_Filter.triggered.connect(self.bilateral_filter)
        self.ui.actionGaussian_Filter.triggered.connect(self.gaussian_filter)

        # Image Enhancement
        self.ui.actionGrayy.triggered.connect(self.Gray)
        self.ui.actionNegative.triggered.connect(self.Negative)
        self.ui.actionHistogram.triggered.connect(self.histogram_Equalization)
        self.ui.actionLog.triggered.connect(self.Log)
        self.ui.actionGamma.triggered.connect(self.gamma)

        # Histogram Plotting
        self.ui.actionHistogram_PDF.triggered.connect(self.hist)

        # Connect the tabChanged signal to the on_tab_changed method
        self.ui.tabWidget.currentChanged.connect(self.on_tab_changed)

        # Set the menu bar initially invisible
        self.ui.menuBar.setVisible(False)

        # Video Processing Buttons
        self.ui.grayscale_btn.clicked.connect(self.start_grayscale)
        self.ui.canny_btn.clicked.connect(self.start_canny)
        self.ui.otsu_btn.clicked.connect(self.start_otsu)
        self.ui.comparison_btn.clicked.connect(self.start_comparison)

        ####################### End IMage Attributes ##########################
        ####################### Start Chart Attributes ##########################

        # Connect the buttons' click events to appropriate methods
        self.ui.btn_plotchat.clicked.connect(self.plot_data)
        self.ui.btn_resetchat.clicked.connect(self.reset_ui)

        # Populate the location (country) ComboBox with your data
        self.populate_countries()

        # Create Matplotlib figure for plotting
        self.figure_chart, self.ax_chart = plt.subplots()
        self.canvas_chart = FigureCanvas(self.figure_chart)
        self.ui.verticalLayout_chart.addWidget(self.canvas_chart)

        ####################### End Chart Attributes ##########################

        ############################ MAP START######################################################

        # Create a QWebEngineView to display the map
        self.web_view = QWebEngineView()
        # self.ui.mapwidget.layout().addWidget(self.web_view)
        self.ui.mapwidget.layout().addWidget(self.web_view)

        # Load the Folium map HTML file into the QWebEngineView
        self.web_view.setHtml(
            "<iframe src='C:/Users/Etudiant FST/PycharmProjects/pythonProject/vaccination_map2.html' width='100%' height='100%' frameborder='0'></iframe>")
        ############################ MAP END ############################################################

        ################### GRAPH Attributes #####################################
        # Buttons Connect
        self.ui.btn_plot_single.clicked.connect(self.plot_single_country)
        self.ui.btn_plot_multi.clicked.connect(self.plot_multi_countries)
        self.ui.btn_reset.clicked.connect(self.reset_plot)

        # Load My Dataset
        self.data = pd.read_csv('C:/Users/Etudiant FST/Desktop/Project/updated_file.csv')

        # Populate both comboBox
        default_item = "Select any country"
        country_list = [default_item] + self.data['location'].unique().tolist()
        self.ui.combo_country.addItems(country_list)
        self.ui.combo_data.addItem("Choose Data to Display")
        self.ui.combo_data.addItems(['daily_vaccinations','total_vaccinations', 'people_vaccinated','daily_people_vaccinated', 'people_fully_vaccinated'])

        # Populate ListView
        self.ui.list_multi_country.addItems(self.data['location'].unique())
        self.ui.list_multi_country.setSelectionMode(QListWidget.MultiSelection)

        # Lineedit for searchable
        self.ui.search_input.textChanged.connect(self.filter_countries)

        # Create Matplotlib figure for plotting
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.ui.verticalLayout_12.addWidget(self.canvas)

############################ End Graph Attributes############################

    ####################### Start Chart Methods ########################

    def populate_countries(self):
        # Load your dataset (e.g., from a CSV file)
        # self.data = pd.read_csv('C:/Users/HP USER/Desktop/Project/updated_file.csv')

        # Extract unique country names from the 'location' column
        unique_countries = self.data['location'].unique()

        # Populate the ComboBox with the unique country names
        self.ui.comboBoxchat.addItem("Select a Country")
        self.ui.comboBoxchat.addItems(unique_countries)

    def plot_data(self):
        try:
            # Retrieve user selections
            selected_country = self.ui.comboBoxchat.currentText()
            selected_data = []
            if self.ui.total_box.isChecked():
                selected_data.append('total_vaccinations')
            if self.ui.people_box.isChecked():
                selected_data.append('people_vaccinated')
            if self.ui.fully_box.isChecked():
                selected_data.append('people_fully_vaccinated')

                # Check if a country is selected
            if selected_country == "Select a Country":
                self.show_error("Select any country.")
                return

            # Check if at least one checkbox is checked
            if not any([self.ui.total_box.isChecked(), self.ui.people_box.isChecked(), self.ui.fully_box.isChecked()]):
                self.show_error("Select any data.")
                return

            # Check if either line or scatter plot is selected
            if not any([self.ui.line_check.isChecked(), self.ui.scatter_check.isChecked()]):
                self.show_error("Select any of the plot (line or scatter).")
                return

            # Check if both line and scatter plot are selected
            if self.ui.line_check.isChecked() and self.ui.scatter_check.isChecked():
                self.show_error("Select either line or scatter plot, not both.")
                return

            # Filter data for the selected country
            filtered_data = self.data[self.data['location'] == selected_country]
            filtered_data = filtered_data.copy()

            # Remove rows with missing values (NaN)
            filtered_data = filtered_data.dropna()

            # Plot line or scatter based on user selections
            if self.ui.line_check.isChecked():
                self.plot_line_chart(selected_country, filtered_data, selected_data,self.ax_chart)
            if self.ui.scatter_check.isChecked():
                self.plot_scatter_chart(selected_country, filtered_data, selected_data, self.ax_chart)

        except Exception as e:
            print("Error:", e)

    def show_error(self, message):
        # Display the error message in the QLabel
        self.ui.labelchat.setText(f"<font color='red'>{message}</font>")

    def plot_line_chart(self, selected_country, filtered_data, selected_data, ax_chart):
        # Check if selected_country and selected_data are not empty
        if selected_country and selected_data:
            selected_data.insert(0, 'date_range')

            # Customize and display your line chart
            for column in selected_data[1:]:
                self.ax_chart.plot(filtered_data['date_range'], filtered_data[column], label=column)
            self.ax_chart.set_xlabel('Date')
            self.ax_chart.set_ylabel('Selected Data')
            self.ax_chart.ticklabel_format(style='plain', axis='y')  # Formatting Y axis
            self.ax_chart.tick_params(axis='x', rotation=40)  # Corrected setting x-axis ticks
            self.ax_chart.tick_params(axis='x', which='major', labelsize=5)
            self.ax_chart.grid(True)
            self.ax_chart.set_title(f'Line Chart for {selected_country}')
            self.ax_chart.legend(loc='upper left')

            self.canvas_chart.draw()

    def plot_scatter_chart(self, selected_country, filtered_data, selected_data, ax_chart):
        # Plot a scatter chart for the selected data
        if selected_country and selected_data:
            for column in selected_data:
                self.ax_chart.scatter(filtered_data['date_range'], filtered_data[column], label=column, marker='o')
            self.ax_chart.set_xlabel('Date')
            self.ax_chart.set_ylabel('Selected Data')
            self.ax_chart.ticklabel_format(style='plain', axis='y')  # Formatting Y axis
            self.ax_chart.tick_params(axis='x', rotation=40)  # Corrected setting x-axis ticks
            # Adjust the font size of the tick labels
            self.ax_chart.tick_params(axis='x', which='major', labelsize=5)

            self.ax_chart.grid(True)
            self.ax_chart.set_title(f'Scatter Plot for {selected_country}')
            self.ax_chart.legend(loc='upper left')

            self.canvas_chart.draw()

    def reset_ui(self):
        # Reset all user selections and clear the display
        self.ui.comboBoxchat.setCurrentIndex(0)
        self.ui.total_box.setChecked(False)
        self.ui.people_box.setChecked(False)
        self.ui.fully_box.setChecked(False)
        self.ui.line_check.setChecked(False)
        self.ui.scatter_check.setChecked(False)
        self.ax_chart.clear()

        self.canvas_chart.draw()
        # Clear the error label
        self.ui.labelchat.clear()


    ####################### End Chart Methods ########################

    def open_url(self, link):
        QDesktopServices.openUrl(QUrl(link))

###################### Start Graph Methods ########################################
    def set_error_message(self, message):
        error_style = "<font color='red' size='5'><b>{}</b></font>".format(message)
        self.ui.label_8.setText(error_style)

    def clear_error_message(self):
        self.ui.label_8.clear()
    def plot_single_country(self):
        country = self.ui.combo_country.currentText()
        selected_data = self.ui.combo_data.currentText()

        if country == "Select any country":
            self.set_error_message("Error: Select any country")
            return

        if selected_data == "Choose Data to Display":
            self.set_error_message("Error: Select any data")
            return

        country_data = self.data[self.data['location'] == country]
        self.ax.clear()
        self.ax.bar(country_data['date_range'], country_data[selected_data], label=country, color='blue')
        self.ax.set_xlabel('Date', fontsize=8)
        self.ax.set_ylabel(selected_data)
        self.ax.ticklabel_format(style='plain', axis='y')
        self.ax.set_title(f'{selected_data} in {country}', fontsize=10)
        plt.xticks(rotation=40)
        self.ax.tick_params(axis='x', which='major', labelsize=6)
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def plot_multi_countries(self):
        selected_countries = [item.text() for item in self.ui.list_multi_country.selectedItems()]
        selected_data = self.ui.combo_data.currentText()

        if not selected_countries:
            self.set_error_message("Error: Select at least one country")
            return

        self.ax.clear()
        for country in selected_countries:
            country_data = self.data[self.data['location'] == country]
            self.ax.bar(country_data['date_range'], country_data[selected_data], label=country)

        self.ax.set_xlabel('Date', fontsize=8)
        self.ax.set_ylabel(selected_data)
        self.ax.ticklabel_format(style='plain', axis='y')
        self.ax.set_title(f'{selected_data} in {", ".join(selected_countries)}', fontsize=10)
        plt.xticks(rotation=40)
        self.ax.tick_params(axis='x', which='major', labelsize=6)
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def reset_plot(self):
        # Set the combo_country to the default item
        default_index = self.ui.combo_country.findText("Select any country")
        self.ui.combo_country.setCurrentIndex(default_index)

        self.ax.clear()
        self.ui.list_multi_country.clearSelection()
        self.canvas.draw()

        # Clear error message
        self.clear_error_message()

    def filter_countries(self):
        text = self.ui.search_input.text().lower()
        self.ui.list_multi_country.clear()

        for country in self.data['location'].unique():
            if text in country.lower():
                item = QListWidgetItem(country)
                self.ui.list_multi_country.addItem(item)

        #################################End Graph Methods###############################################




############################ Start Image Methods####################################################

##########################__TAB__#########################################################

    def start_grayscale(self):
        self.timer.start(30)
        self.timer.timeout.connect(self.show_grayscale)

    def show_grayscale(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Grayscale Frame', gray)
    def start_canny(self):
        self.timer.start(30)
        self.timer.timeout.connect(self.show_canny)

    def show_canny(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            cv2.imshow('Canny Edge Detection', edges)

    def start_otsu(self):
        self.timer.start(30)
        self.timer.timeout.connect(self.show_otsu)

    def show_otsu(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow('Otsu Thresholding', thresh1)

    def start_comparison(self):
        self.timer.start(30)
        self.timer.timeout.connect(self.show_comparison)

    def show_comparison(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            cv2.imshow('Original Frame', frame)
            cv2.imshow('Grayscale Frame', gray)
            cv2.imshow('Edges', edges)
    def on_tab_changed(self, index):
        # Check if Tab2 is selected
        if index == 1:
            # Set the menu bar visible for Tab2
            self.ui.menuBar.setVisible(True)
        else:
            # Set the menu bar invisible for other tabs
            self.ui.menuBar.setVisible(False)

    @pyqtSlot()
    def loadImage(self, fname):
        try:
            self.ui.image = cv2.imread(fname)
            if self.ui.image is not None:
                self.ui.tmp = self.ui.image
                self.displayImage()
                # Show a label indicating successful image upload
                success_message = "Image Successfully Upload"

                # Set color, font size, and alignment using style sheet
                self.ui.label_12.setStyleSheet(f"color: green; font-size: 12px;font-weight: bold ;text-align: center;")
                self.ui.label_12.setText(success_message)
            else:
                print("Error loading image:", fname)
        except Exception as e:
            print("Error loading image:", e)

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8
        if len(self.ui.image.shape) == 3:
            if self.ui.image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(
            self.ui.image.data, self.ui.image.shape[1], self.ui.image.shape[0], self.ui.image.strides[0], qformat)

        img = img.rgbSwapped()
        if window == 1:
            self.ui.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.ui.imgLabel.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window == 2:
            self.ui.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.ui.imgLabel2.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def open_img(self):
        fname, filter = QFileDialog.getOpenFileName(
            self, 'Open File', 'C:\\', "Image Files (*)")
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")

    def save_img(self):
        fname, filter = QFileDialog.getSaveFileName(
            self, 'Save File', 'C:\\', "Image Files (*.png)")
        if fname:
            cv2.imwrite(fname, self.ui.image)
            print("Error")

    def createPrintDialog(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.ui.imgLabel2.print_(printer)

    def big_Img(self):
        self.ui.image = cv2.resize(self.ui.image, None, fx=1.5,fy=1.5, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def small_Img(self):
        self.ui.image = cv2.resize(self.ui.image, None, fx=0.75,fy=0.75, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)



    def reset(self):
        try:
            if self.ui.tmp is not None:
                self.ui.image = self.ui.tmp
                self.displayImage(2)

                # Clear error message
                self.ui.label_12.clear()

            else:
                print("Error: No image to reset.")
        except Exception as e:
            print("Error resetting image:", e)

    def QuestionMessage(self):
        message = QMessageBox.question(
            self, "Exit", "Are you sure you want to exit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

############################# Transformations ##############################################

    def rotation(self, angle):
        if self.ui.tmp is not None and self.ui.tmp.size > 0:
            rows, cols, steps = self.ui.tmp.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_image = cv2.warpAffine(self.ui.tmp, M, (cols, rows))
            self.ui.image = rotated_image
            self.displayImage(2)

    def shearing(self):
        self.ui.image = self.ui.tmp
        rows, cols, ch = self.ui.image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        self.ui.image = cv2.warpAffine(self.ui.image, M, (cols, rows))
        self.displayImage(2)

    def translation(self):
        self.ui.image = self.ui.tmp
        num_rows, num_cols = self.ui.image.shape[:2]

        translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
        img_translation = cv2.warpAffine(
            self.ui.image, translation_matrix, (num_cols, num_rows))
        self.ui.image = img_translation
        self.displayImage(2)

    def erode(self, iter):
        self.ui.image = self.ui.tmp
        if iter > 0:
            kernel = np.ones((4, 7), np.uint8)
            self.ui.image = cv2.erode(self.ui.tmp, kernel, iterations=iter)
        else:
            kernel = np.ones((2, 6), np.uint8)
            self.ui.image = cv2.dilate(self.ui.image, kernel, iterations=iter * -1)
        self.displayImage(2)

    def Canny(self):
        self.ui.image = self.ui.tmp
        if self.ui.canny.isChecked():
            can = cv2.cvtColor(self.ui.image, cv2.COLOR_BGR2GRAY)
            self.ui.image = cv2.Canny(
                can, self.ui.canny_min.value(), self.ui.canny_max.value())
        self.displayImage(2)

 ################################ Image Enhancement #################################################################
    def Gray(self):
        self.ui.image = self.ui.tmp
        self.ui.image = cv2.cvtColor(self.ui.image, cv2.COLOR_BGR2GRAY)
        self.displayImage(2)


    def Negative(self):
        self.ui.image = self.ui.tmp
        self.ui.image = cv2.bitwise_not(self.ui.image)
        self.displayImage(2)

    def histogram_Equalization(self):
        self.ui.image = self.ui.tmp
        img_yuv = cv2.cvtColor(self.ui.image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        self.ui.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        self.displayImage(2)

    def Log(self):
        self.ui.image = self.ui.tmp
        img_2 = np.uint8(np.nan_to_num(np.log(np.maximum(1, self.ui.image))))

        c = 2
        self.ui.image = cv2.threshold(img_2, c, 225, cv2.THRESH_BINARY)[1]
        self.displayImage(2)

    def Gamma_(self, gamma):
        self.ui.image = self.ui.tmp
        gamma = gamma * 0.1
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        self.ui.image = cv2.LUT(self.ui.image, table)
        self.displayImage(2)

    def gamma(self):
        self.ui.image = self.ui.tmp
        gamma = 2.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        self.ui.image = cv2.LUT(self.ui.image, table)
        self.displayImage(2)

    # ####################################### Image Restoration ################################################################
    def hist(self):
        self.ui.image = self.ui.tmp
        histg = cv2.calcHist([self.ui.image], [0], None, [256], [0, 256])
        self.ui.image = histg
        plt.plot(self.ui.image)
        plt.show()
        self.displayImage(2)

 ################################## Simple Edge Detection #################################################################
    def hough_transform_edge_detection(self):
        gray = cv2.cvtColor(self.ui.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(self.ui.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.displayImage(2)

    def directional_filtering(self):
        self.ui.image = self.ui.tmp
        kernel = np.ones((3, 3), np.float32) / 9
        self.ui.image = cv2.filter2D(self.ui.image, -1, kernel)
        self.displayImage(2)

    ##################################### Filter ##########################################################################
    def blur(self):
        self.ui.image = self.ui.tmp
        self.ui.image = cv2.blur(self.ui.image, (5, 5))
        self.displayImage(2)

    def box_filter(self):
        self.ui.image = self.ui.tmp
        self.ui.image = cv2.boxFilter(self.ui.image, -1, (20, 20))
        self.displayImage(2)

    def median_filter(self):
        self.ui.image = self.ui.tmp
        self.ui.image = cv2.medianBlur(self.ui.image, 5)
        self.displayImage(2)

    def bilateral_filter(self):
        self.ui.image = self.ui.tmp
        self.ui.image = cv2.bilateralFilter(self.ui.image, 9, 75, 75)
        self.displayImage(2)

    def gaussian_filter(self):
        self.ui.image = self.ui.tmp
        self.ui.image = cv2.GaussianBlur(self.ui.image, (5, 5), 0)
        self.displayImage(2)

    def gaussian_filter2(self, g):
        self.ui.image = self.ui.tmp
        self.ui.image = cv2.GaussianBlur(self.ui.image, (5, 5), g)
        self.displayImage(2)

if __name__ == "__main__":
    app = QApplication([])
    window = MyMainWindow()
    window.show()
    app.exec_()