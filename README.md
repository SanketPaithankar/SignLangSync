# *Sign Language Recognition Using MediaPipe and LSTM model*

**This project demonstrates the use of MediaPipe Holistic for recognizing sign language gestures. The project leverages keypoint detection and visualization techniques to extract and analyze landmarks from video frames captured through a webcam.**

## **Table of Contents**

* [Introduction](#introduction)
* [Installation](#installation)
* [Usage](#usage)
* [Functions Overview](#functions-overview)
* [Visualization](#visualization)
* [Extracting Keypoints](#extracting-keypoints)
* [Saving and Loading Keypoints](#saving-and-loading-keypoints)
* [Acknowledgements](#acknowledgements)

## **Introduction**

*The goal of this project is to detect and visualize keypoints from video frames to recognize sign language gestures. This is achieved using MediaPipe Holistic, which provides an integrated solution for face, hand, and pose landmark detection.*

## **Installation**

**To install the necessary packages, run the following commands:**

```sh
pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 
pip install opencv-python 
pip install mediapipe 
pip install scikit-learn 
pip install matplotlib
```

## **Usage**

**To use this project, run the provided Jupyter notebook. The notebook includes all the necessary code for capturing video frames, detecting keypoints, and visualizing the results.**

## **Functions Overview**

### *1. Import and Install Dependencies*

**This section installs and imports the required libraries:**

* **TensorFlow 2.4.1 and TensorFlow-GPU 2.4.1**: Essential for running deep learning models. TensorFlow-GPU allows the use of GPU acceleration to speed up computations. If you don’t have a dedicated GPU, you can skip this step.
* **OpenCV**: Used for computer vision tasks such as image and video processing.
* **MediaPipe**: A framework for building multimodal (e.g., video, audio) applied machine learning pipelines.
* **Scikit-learn**: A machine learning library for Python, providing simple and efficient tools for data mining and data analysis.
* **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.

### *2. Keypoints using MediaPipe Holistic*

**This section initializes the MediaPipe Holistic model and defines functions for detecting and drawing landmarks on video frames:**

* **mediapipe_detection**: Converts the image color format, processes it using the MediaPipe model, and then converts it back.
* **draw_landmarks**: Draws detected landmarks on the image.
* **draw_styled_landmarks**: Draws landmarks with customized styles for better visualization.

### *3. Visualizing Keypoints*

**This section demonstrates how to draw detected landmarks on a frame and visualize the results using Matplotlib:**

```python
draw_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

### *4. Extract Keypoint Values*

**This section extracts the keypoint values from the detected landmarks:**

```python
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
```

### *5. Saving and Loading Keypoints*

**This section demonstrates how to save and load the extracted keypoints using NumPy:**

```python
np.save('0', result_test)
np.load('0.npy')
```

## **Acknowledgements**

*This project utilizes the following libraries and frameworks:*

* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [MediaPipe](https://mediapipe.dev/)
* [Scikit-learn](https://scikit-learn.org/)
* [Matplotlib](https://matplotlib.org/)

*Here are some output Screenshots*

![Screenshot 2024-04-19 083806](https://github.com/user-attachments/assets/4e56ec70-0022-4409-9ff8-640793f4fe06)
![Screenshot 2024-04-19 083735](https://github.com/user-attachments/assets/395ab928-f3c2-4291-b4aa-15d472962ff6)
![Screenshot 2024-04-19 083822](https://github.com/user-attachments/assets/ba2376a4-e79f-43c1-87a4-a90c2bd6e2f5)

*Feel free to explore and modify the notebook to fit your specific needs. Happy coding!*
