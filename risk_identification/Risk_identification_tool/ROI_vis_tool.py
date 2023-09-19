import argparse
import sys

from PyQt5 import QtWidgets
from utils.controller import MainWindow_controller


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', default="/PATH/TO/DATASET/RiskBench_Dataset", type=str)
    parser.add_argument('--metadata_root', default="./metadata", type=str)
    parser.add_argument('--model_root', default="./model", type=str)
    parser.add_argument('--vis_result_path',
                        default="./ROI_vis_result", type=str)
    parser.add_argument('--FPS', default=20, type=int)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller(args)
    window.show()
    sys.exit(app.exec_())
