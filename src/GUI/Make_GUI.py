"""
This program uses pyuic to transform the "xxx.ui" file into "xxx.py" file.
"""
import subprocess
from os import path

directory = path.dirname(path.abspath(__file__))
try:
    subprocess.call(["pyuic5", path.join(directory, "ui_files\\MainWindow.ui"), ">",
                     path.join(directory, "ui_py\\MainWindowUI.py")], shell=True)
except BaseException as e:
    print(e)

