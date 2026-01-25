
from roboflow import Roboflow
rf = Roboflow(api_key="n5dCC32yOxQH3RUHSxfo")
project = rf.workspace("nawneet").project("fishy-v0get")
version = project.version(3)
dataset = version.download("yolov8")
                