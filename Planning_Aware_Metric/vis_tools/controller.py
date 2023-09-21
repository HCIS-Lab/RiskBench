from PyQt5 import QtCore 
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
# from PyQt5.QtCore import QThread, pyqtSignal

import time
import os


from UI import Ui_MainWindow
from video_controller import video_controller
import json

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # os 
        self.root = "../data_collection"
        scenario_list = [  "obstacle", "interactive", "non-interactive", "collision" ]
        scenario_list_check = os.listdir(self.root)
        scenario_types = []
        
        # check scenario exists
        for scenario in scenario_list:
            if scenario in scenario_list_check:
                scenario_types.append(scenario)
        
        for scenario in scenario_types:
            self.ui.comboBox_scenario.addItem(scenario)
        self.scenario = self.ui.comboBox_scenario.currentText()
        
        for folder_name in sorted(os.listdir(f"{self.root}/{self.scenario}")):
            if folder_name != ".DS_Store" :
                self.ui.comboBox_basic_scenario.addItem(folder_name)
        self.current_folder = self.ui.comboBox_basic_scenario.currentText()
        self.folder_path = f"{self.root}/{self.scenario}/{self.current_folder}/"
        self.video_path = f"{self.root}/{self.scenario}/{self.current_folder}/{self.current_folder}.mp4"
        self.load_json()
        
        self.video_controller = video_controller(video_path=self.video_path, ui=self.ui)
        self.ui.button_play.clicked.connect(self.video_controller.play) # connect to function()
        self.ui.button_stop.clicked.connect(self.video_controller.stop)
        self.ui.button_pause.clicked.connect(self.video_controller.pause)
        self.ui.pushButton_next.clicked.connect(self.variant_next)
        self.ui.pushButton_back.clicked.connect(self.variant_back)
        self.ui.comboBox_scenario.currentIndexChanged.connect(self.update_scenario)
        self.ui.comboBox_basic_scenario.currentIndexChanged.connect(self.update_basic)
        self.ui.save_json.clicked.connect(self.write_json)
        
    def load_json(self):
        # check file 
        self.ui.start_x.clear()
        self.ui.start_y.clear()
        self.ui.end_x.clear()
        self.ui.end_y.clear()
        
        if os.path.exists(f"{self.folder_path}start_end_point.json"):
            with open(f"{self.folder_path}start_end_point.json", 'r') as openfile:
                data = json.load(openfile)
                self.ui.start_x.setText(str(data["start_x"]))
                self.ui.start_y.setText(str(data["start_y"]))
                self.ui.end_x.setText(str(data["end_x"]))
                self.ui.end_y.setText(str(data["end_y"]))
            
    def write_json(self):
        data = {
            "start_x": float(self.ui.start_x.text()),
            "start_y": float(self.ui.start_y.text()),
            "end_x": float(self.ui.end_x.text()),
            "end_y": float(self.ui.end_y.text()),
        }
        json_object = json.dumps(data, indent=4)
        with open(f"{self.folder_path}start_end_point.json", "w") as outfile:
            outfile.write(json_object)
        
    def update_basic(self):
        if self.ui.comboBox_basic_scenario.count()!= 0:
            self.update_path()
        
    def update_scenario(self):
        self.scenario = self.ui.comboBox_scenario.currentText()
        self.ui.comboBox_basic_scenario.clear()
        for folder_name in sorted(os.listdir(f"{self.root}/{self.scenario}")):
            if folder_name != ".DS_Store":
                self.ui.comboBox_basic_scenario.addItem(folder_name)
        self.current_folder = self.ui.comboBox_basic_scenario.currentText()
        self.folder_path = f"{self.root}/{self.scenario}/{self.current_folder}/"
        self.video_path = f"{self.root}/{self.scenario}/{self.current_folder}/{self.current_folder}.mp4"
        
        # for folder_name in sorted(os.listdir(f"{self.root}/{self.scenario}")):
        #     if folder_name != ".DS_Store":
        #         self.ui.comboBox_basic_scenario.addItem(folder_name)
            
        self.update_path()
        
    def variant_back(self):
        index = self.ui.comboBox_basic_scenario.currentIndex()
        index -=1
        if index < 0: 
            index = self.ui.comboBox_basic_scenario.count() - 1
        self.ui.comboBox_basic_scenario.setCurrentIndex(index)
        self.update_path()
        
    def variant_next(self):
        index = self.ui.comboBox_basic_scenario.currentIndex()
        index +=1
        if index >= self.ui.comboBox_basic_scenario.count():
            index = 0
        self.ui.comboBox_basic_scenario.setCurrentIndex(index)
        self.update_path()
        
    def update_path(self):
        self.current_folder = self.ui.comboBox_basic_scenario.currentText()
        self.folder_path = f"{self.root}/{self.scenario}/{self.current_folder}/"
        self.video_path = f"{self.root}/{self.scenario}/{self.current_folder}/{self.current_folder}.mp4"
        self.video_controller.update_video_path(self.video_path)
        self.load_json()
        
        