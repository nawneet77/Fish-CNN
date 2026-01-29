

from roboflow import Roboflow
rf = Roboflow(api_key="n5dCC32yOxQH3RUHSxfo")
project = rf.workspace("nawneet").project("fishcnn")
version = project.version(1)
dataset = version.download("yolov8")
                