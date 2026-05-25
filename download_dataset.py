from roboflow import Roboflow
rf = Roboflow(api_key="R7gl4oUVcXj9asx4IChO")
project = rf.workspace("bhanu-prasad-vlzga").project("person-detection-lds4b")
version = project.version(2)
dataset = version.download("yolov11")
                