import os
import json
import time

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from .UI import Ui_MainWindow
from .video_controller import video_controller
from .cal_metric import ROI_evaluation
from .utils import filter_roi_scenario, read_metadata, get_scenario_info, make_video


class MainWindow_controller(QMainWindow):
    def __init__(self, args):

        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.data_root = args.data_root
        self.metadata_root = args.metadata_root
        self.model_root = args.model_root
        self.vis_result_path = args.vis_result_path
        self.FPS = args.FPS

        self.behavior_dict = None
        self.gt_risk_dict = None
        self.critical_dict = None

        self.model_list = ["Random", "Range", "Kalman filter", "Social-GAN",
                           "MANTRA", "QCNet", "DSA", "RRL", "BP", "BCP"]
        self.type_list = ["interactive",
                          "non-interactive", "obstacle", "collision"]

        self.scenario_list = []
        self.user_attr_list = []
        self.roi_result = None
        self.metric_result = None
        self.pre_metric_result = None

        self.model_clear = False
        self.type_clear = False
        self.scenario_clear = False

        self.model_change = True
        self.type_change = True
        self.scenario_change = True

        self.update_model()
        self.update_type()

        self.video_controller = video_controller(ui=self.ui, FPS=self.FPS)
        self.ui.button_play.clicked.connect(self.video_controller.play)
        self.ui.button_pause.clicked.connect(self.video_controller.pause)

        self.ui.button_next.clicked.connect(self.scenario_next)
        self.ui.button_back.clicked.connect(self.scenario_back)
        self.ui.button_filter_scenario.clicked.connect(self.filter_scenario)
        self.ui.button_gen_gif.clicked.connect(self.gen_gif)
        self.ui.button_gen_json.clicked.connect(self.gen_json)

        self.ui.comboBox_model.currentIndexChanged.connect(
            self.update_model)
        self.ui.comboBox_type.currentIndexChanged.connect(
            self.update_type)
        self.ui.comboBox_scenario_list.currentIndexChanged.connect(
            self.update_scenario)

        self.checkBox_list = {
            "actor": [
                self.ui.checkBox_actor_all,
                self.ui.checkBox_actor_car,
                self.ui.checkBox_actor_truck,
                self.ui.checkBox_actor_bike,
                self.ui.checkBox_actor_motor,
                self.ui.checkBox_actor_pedestrian,
                self.ui.checkBox_actor_2wheel,
                self.ui.checkBox_actor_4wheel,
                self.ui.checkBox_actor_obstacle,
                self.ui.checkBox_actor_parking,

            ],
            "weather": [
                self.ui.checkBox_weather_all,
                self.ui.checkBox_weather_clear,
                self.ui.checkBox_weather_cloudy,
                self.ui.checkBox_weather_wet,
                self.ui.checkBox_weather_rain,
            ],
            "time": [
                self.ui.checkBox_time_all,
                self.ui.checkBox_time_noon,
                self.ui.checkBox_time_sunset,
                self.ui.checkBox_time_night,
            ],
            "density": [
                self.ui.checkBox_density_all,
                self.ui.checkBox_density_low,
                self.ui.checkBox_density_mid,
                self.ui.checkBox_density_high,
            ],
            "topology": [
                self.ui.checkBox_topology_all,
                self.ui.checkBox_topology_4way,
                self.ui.checkBox_topology_3way_A,
                self.ui.checkBox_topology_3way_B,
                self.ui.checkBox_topology_3way_C,
                self.ui.checkBox_topology_straight,
                self.ui.checkBox_topology_roundabout,
            ]}

        def init_checkBox(checkBox, attr_type, i):
            checkBox.clicked.connect(
                lambda: self.update_attribute(checkBox, attr_type, i))

        for attr_type in self.checkBox_list:
            for i in range(len(self.checkBox_list[attr_type])):
                init_checkBox(self.checkBox_list[attr_type][i], attr_type, i)

    def update_model(self):

        if self.model_change:

            if not self.model_clear:

                self.model_clear = True
                self.model_change = False

                self.ui.comboBox_model.clear()

                for model in self.model_list:
                    self.ui.comboBox_model.addItem(model)

                self.model_change = True

    def update_type(self):

        if self.type_change:

            if not self.type_clear:
                self.type_clear = True
                self.type_change = False

                self.ui.comboBox_type.clear()

                for _type in self.type_list:
                    self.ui.comboBox_type.addItem(_type)

                self.type_change = True

            data_type = self.ui.comboBox_type.currentText()
            self.behavior_dict, self.gt_risk_dict, self.critical_dict = read_metadata(
                data_type, self.metadata_root)

            self.scenario_list = []
            self.scenario_clear = False
            self.update_scenario()

    def update_scenario(self):

        font = QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(12)

        if self.scenario_change and not self.scenario_clear:

            self.scenario_clear = True
            self.scenario_change = False

            self.ui.comboBox_scenario_list.clear()

            self.ui.comboBox_scenario_list.addItem("Selected Scenario")
            font.setBold(True)
            self.ui.comboBox_scenario_list.model().item(0).setFont(font)
            self.ui.comboBox_scenario_list.setFont(font)

            for i, scenario in enumerate(self.scenario_list, 1):
                self.ui.comboBox_scenario_list.addItem(scenario)
                font.setBold(False)
                self.ui.comboBox_scenario_list.model().item(i).setFont(font)

            self.ui.label_scenario_cnt.setText(
                f"0/{self.ui.comboBox_scenario_list.count()-1}")

            self.scenario_change = True

        elif self.scenario_change:

            cur_idx = self.ui.comboBox_scenario_list.currentIndex()
            self.ui.label_scenario_cnt.setText(
                f"{cur_idx}/{self.ui.comboBox_scenario_list.count()-1}")

            if cur_idx == 0:
                font.setBold(True)
            else:
                font.setBold(False)
            self.ui.comboBox_scenario_list.setFont(font)

    def update_attribute(self, checkBox, attr_type=None, i=-1):

        # i==0 means 'All'
        if i == 0 and checkBox.isChecked():
            for attr_checkBox in self.checkBox_list[attr_type]:
                attr_checkBox.setChecked(True)

        elif i != 0 and not checkBox.isChecked():
            self.checkBox_list[attr_type][0].setChecked(False)

    def update_video(self, gif_path):

        assert os.path.exists(gif_path), f"{gif_path} not found"
        self.video_controller.update_video_path(gif_path)

    def scenario_next(self):
        N = self.ui.comboBox_scenario_list.count()

        if N == 1:
            print("No Scenario!")
        else:
            cur_idx = self.ui.comboBox_scenario_list.currentIndex()
            self.ui.comboBox_scenario_list.setCurrentIndex((cur_idx+1) % N)

    def scenario_back(self):

        N = self.ui.comboBox_scenario_list.count()

        if N == 1:
            print("No Scenario!")
        else:
            cur_idx = self.ui.comboBox_scenario_list.currentIndex()
            self.ui.comboBox_scenario_list.setCurrentIndex((cur_idx-1) % N)

    def filter_scenario(self):

        data_type = self.ui.comboBox_type.currentText()
        model = self.ui.comboBox_model.currentText()
        self.user_attr_list = []

        roi_result = {}

        for i, attr_type in enumerate(list(self.checkBox_list.keys())):
            attr_list = []

            for attr_checkBox in self.checkBox_list[attr_type]:
                attr = attr_checkBox.text()
                if attr != "All" and attr_checkBox.isChecked():
                    attr_list.append(attr)
                    self.user_attr_list.append(attr)

            attr_roi_result = filter_roi_scenario(
                data_type, model, attr_list, self.model_root)

            if (data_type == "non-interactive" and i == 1) or i == 0:
                roi_result = attr_roi_result
            else:
                roi_result = {key: roi_result[key] for key in set(
                    roi_result.keys()).intersection(set(attr_roi_result.keys()))}

        self.scenario_list = sorted(list(roi_result.keys()))
        self.roi_result = roi_result

        self.scenario_clear = False
        self.update_scenario()

        if len(roi_result) == 0:
            self.show_no_result()
        else:
            self.show_result()

    def gen_gif(self):

        if self.ui.comboBox_scenario_list.count() == 1:
            print("No Scenario!")
            return
        elif self.ui.comboBox_scenario_list.currentIndex() == 0:
            print("Please select scenario!!")
            return

        scenario_name = self.ui.comboBox_scenario_list.currentText()
        data_type = self.ui.comboBox_type.currentText()
        model = self.ui.comboBox_model.currentText()

        basic, variant, roi, behavior, risky_id = get_scenario_info(scenario_name,
                                                                    data_type, self.roi_result, self.behavior_dict, self.gt_risk_dict)
        variant_path = os.path.join(
            self.data_root, data_type, basic, "variant_scenario", variant)

        start = time.time()
        print(f"Generating video  '{scenario_name}' ...")

        gif_save_folder = os.path.join(
            self.vis_result_path, "gif", model, data_type, basic)
        if not os.path.isdir(gif_save_folder):
            os.makedirs(gif_save_folder)
        gif_save_path = os.path.join(gif_save_folder, f"{variant}.gif")

        make_video(gif_save_path, variant_path,
                   roi, behavior, risky_id, self.FPS)

        self.update_video(gif_save_path)

        end = time.time()
        print(f"Generated '{gif_save_path}' in {end-start:.2f} secs\n")

        self.video_controller.play()

    def gen_json(self):

        if self.ui.comboBox_scenario_list.count() == 1:
            print("No Scenario!")
            return

        data_type = self.ui.comboBox_type.currentText()
        model = self.ui.comboBox_model.currentText()

        json_save_folder = os.path.join(
            self.vis_result_path, "json", model, data_type)
        if not os.path.isdir(json_save_folder):
            os.makedirs(json_save_folder)

        json_save_path = os.path.join(json_save_folder, f"roi_result.json")

        with open(json_save_path, "w") as f:
            json.dump(self.metric_result, f, indent=4)

        print(f"Generated '{json_save_path}'\n")

    def show_result(self):

        print(f"There are {len(self.roi_result):3d} scenarios.")

        def set_color(label, diff, text, reverse_color=False):

            self.text_color = QGraphicsColorizeEffect()
            if (diff >= 0) ^ reverse_color:
                self.text_color.setColor(Qt.darkRed)
            else:
                self.text_color.setColor(Qt.darkGreen)
            color = self.text_color

            label.setGraphicsEffect(color)
            label.setText(text)

        data_type = self.ui.comboBox_type.currentText()
        model = self.ui.comboBox_model.currentText()

        self.metric_result, _ = ROI_evaluation(data_type, model, self.roi_result,
                                               self.behavior_dict, self.gt_risk_dict, self.critical_dict, attribute=self.user_attr_list)

        self.ui.f1_label.setText(
            f"{'F1-Score':9s} : {self.metric_result['f1-Score']*100:.1f}%")
        self.ui.recall_label.setText(
            f"{'Recall':9s} : {self.metric_result['recall']*100:.1f}%")
        self.ui.precision_label.setText(
            f"{'Precision':9s} : {self.metric_result['precision']*100:.1f}%")

        if self.pre_metric_result != None:

            diff_f1 = self.metric_result['f1-Score'] - \
                self.pre_metric_result['f1-Score']
            diff_recall = self.metric_result['recall'] - \
                self.pre_metric_result['recall']
            diff_precision = self.metric_result['precision'] - \
                self.pre_metric_result['precision']

            set_color(self.ui.f1_diff_label, diff_f1, f"({diff_f1*100:+.1f}%)")
            set_color(self.ui.recall_diff_label, diff_recall,
                      f"({diff_recall*100:+.1f}%)")
            set_color(self.ui.precision_diff_label, diff_precision,
                      f"({diff_precision*100:+.1f}%)")

        if data_type == "non-interactive":
            self.ui.consistency_label_1.setText(
                f"{'Consistency 1s':9s} : None")
            self.ui.consistency_label_2.setText(
                f"{'Consistency 2s':9s} : None")
            self.ui.consistency_label_3.setText(
                f"{'Consistency 3s':9s} : None")
            self.ui.pic_fa_label.setText(
                f"{'FA':9s} : {self.metric_result['FA']*100:.1f}%")

            self.ui.consistency_diff_label_1.setText(f"")
            self.ui.consistency_diff_label_2.setText(f"")
            self.ui.consistency_diff_label_3.setText(f"")

            if self.pre_metric_result == None or self.pre_metric_result['type'] != "non-interactive":
                self.ui.pic_fa_diff_label.setText(f"")
            else:
                diff_fa = self.metric_result['FA']-self.pre_metric_result['FA']
                set_color(self.ui.pic_fa_diff_label, diff_fa,
                          f"({diff_fa*100:+.1f}%)", reverse_color=True)

        else:
            self.ui.consistency_label_1.setText(
                f"{'Consistency 1s':9s} : {self.metric_result['Consistency_1s']*100:.1f}%")
            self.ui.consistency_label_2.setText(
                f"{'Consistency 2s':9s} : {self.metric_result['Consistency_2s']*100:.1f}%")
            self.ui.consistency_label_3.setText(
                f"{'Consistency 3s':9s} : {self.metric_result['Consistency_3s']*100:.1f}%")
            self.ui.pic_fa_label.setText(
                f"{'PIC':9s} : {self.metric_result['PIC']:.1f}")

            if self.pre_metric_result == None or self.pre_metric_result['type'] == "non-interactive":
                self.ui.consistency_diff_label_1.setText(f"")
                self.ui.consistency_diff_label_2.setText(f"")
                self.ui.consistency_diff_label_3.setText(f"")
                self.ui.pic_fa_diff_label.setText(f"")
            else:

                diff_consistency_1 = self.metric_result['Consistency_1s'] - \
                    self.pre_metric_result['Consistency_1s']
                diff_consistency_2 = self.metric_result['Consistency_2s'] - \
                    self.pre_metric_result['Consistency_2s']
                diff_consistency_3 = self.metric_result['Consistency_3s'] - \
                    self.pre_metric_result['Consistency_3s']
                diff_pic = self.metric_result['PIC'] - \
                    self.pre_metric_result['PIC']

                set_color(self.ui.consistency_diff_label_1,
                          diff_consistency_1, f"({diff_consistency_1*100:+.1f}%)")
                set_color(self.ui.consistency_diff_label_2,
                          diff_consistency_2, f"({diff_consistency_2*100:+.1f}%)")
                set_color(self.ui.consistency_diff_label_3,
                          diff_consistency_3, f"({diff_consistency_3*100:+.1f}%)")
                set_color(self.ui.pic_fa_diff_label, diff_pic,
                          f"({diff_pic:+.1f})", reverse_color=True)

        self.pre_metric_result = self.metric_result

    def show_no_result(self):

        print("No Scenario!")

        self.ui.f1_label.setText(f"{'F1-Score':9s} : None")
        self.ui.recall_label.setText(f"{'Recall':9s} : None")
        self.ui.precision_label.setText(f"{'Precision':9s} : None")

        self.ui.consistency_label_1.setText(f"{'Consistency 1s':9s} : None")
        self.ui.consistency_label_2.setText(f"{'Consistency 2s':9s} : None")
        self.ui.consistency_label_3.setText(f"{'Consistency 3s':9s} : None")
        self.ui.pic_fa_label.setText(f"{'FA':9s} : None")

        self.ui.f1_diff_label.setText(f"")
        self.ui.recall_diff_label.setText(f"")
        self.ui.precision_diff_label.setText(f"")
        self.ui.consistency_diff_label_1.setText(f"")
        self.ui.consistency_diff_label_2.setText(f"")
        self.ui.consistency_diff_label_3.setText(f"")
        self.ui.pic_fa_diff_label.setText(f"")

        self.pre_metric_result = None
