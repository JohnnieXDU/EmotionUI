# EmotionUI
 Software for Multimodalty 2D+3D Facial Expression Recognition (FER) UI.

## demo screenshot
![demo-ui](https://github.com/JohnnieXDU/EmotionUI/blob/main/resource/demo-ui.png)

![demo-happy](https://github.com/JohnnieXDU/EmotionUI/blob/main/resource/demo-happy.png)

(with RealSense)

## required packages
 - Python >= 3.6
 - numpy >= 1.19.5
 - Opencv-python >= 4.5
 - PySide6 >= 6.2.1
 - PyTorch >= 1.10
 - TorchVision >= 0.11

## loding weights
 Two ways, try them, depending on your Internet speed.
 1) Manually download.
    Download vgg16 weights from: 
    
    https://drive.google.com/file/d/1f-tKgovJ54l9xR3oIZ6gy77NdPirUddr/view?usp=sharing
    
    Then, move the weights to "./weights" folder.
    
 or 
 2) Run script.
    Open terminal, run:
    > cd weights
    > python download_weights.py

## hardware
 1) usb camera
 2) Intel RealSense (optional for depth imaging)

## notice
 The current FER algorithm is vgg16 for simplicity. One can easily change the network as you wish.
