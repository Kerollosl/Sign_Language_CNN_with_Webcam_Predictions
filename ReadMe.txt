Sign Language CNN and HTML Webcam Prediction Readme File
 
Author:
	Kerollos Lowandy

Environment
	OS: Windows
	Programming Language: Python 3.8.2

*Project Adapted from DeepLearning.ai TensorFlow Developer Certificate and TensorFlow Data dn Deployment Certificate Labs*

*Accuracy of HTML Webcam predictions is currently not at viable level. Project is currently a proof of concept*

Instructions

Files Needed in Common Directory:
	- functions.py
	- convert_to_json.py
	- categorical_classification_sign_language.py
	- sign_mnist_train.csv (Can be downloaded from https://www.kaggle.com/datasets/datamunge/sign-language-mnist?select=sign_mnist_train)
	- sign_mnist_test.csv (Can be downloaded from https://www.kaggle.com/datasets/datamunge/sign-language-mnist?select=sign_mnist_test)

Steps:
	1. Run categorical_classification_sign_language.py to generate a model.h5 file.
        2. Run convert_to_json.py to convert the model.h5 file to a model.json file along with the associated bin files. (This currently must be done with a notebook IDE 
	   such as Jupyter Notebook or Google Colab)
	3. Mode the Converted Files to the "Sign Language HTML" directory. Ensure that the 
	   model.json file name matches that referenced in the index.js file in this 
	   directory.
	4. Run the "homepage" HTML file with an Internet server to generate live webcam 
	   image based predictions.

	IDE: PyCharm, VS Code
	Packages Used: 
			tensorflow (2.15.0)
			tensorflowjs (4.16.0)
			matplotlib (3.8.2)
			opencv-python (4.9.0.80)
			numpy (1.26.2)
			random (built into Python 3.8.2)
			csv (built into Python 3.8.2)
			
