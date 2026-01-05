from roboflow import Roboflow
rf = Roboflow(api_key="SJteQbdvJ6AuMeSBI8ec")
project = rf.workspace("techmrt").project("pavement-feature")
version = project.version(2)

# This downloads the actual weights into your current folder
version.download("yolov11")