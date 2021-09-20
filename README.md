# Pytorchを使用したgoogledrive内のファイルを物体検知するプログラム

This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implementedin PyTorch. The code is based on the official code of [YOLO v3](https://github.com/pjreddie/darknet), as well as a PyTorch 
port of the original code, by [marvis](https://github.com/marvis/pytorch-yolo2). One of the goals of this code is to improve
upon the original port by removing redundant parts of the code (The official code is basically a fully blown deep learning 
library, and includes stuff like sequence models, which are not used in YOLO). I've also tried to keep the code minimal, and 
document it as well as I can. 

## Requirements
1. Python 3.6
2. OpenCV
3. PyTorch
4. Pydrive

##実行する際はこちら
setting.yamlにトークン等を入れておくこと
```
python googledrive.py 
```




