# Deepfake Detection App

This repository contains a Gradio-based web application that detects whether an image is a deepfake. The application uses a pre-trained InceptionResnetV1 model from the `facenet_pytorch` library for face recognition, and `pytorch-grad-cam` for visual explainability.

## How It Works

1. **Face Detection**: The application uses the `MTCNN` model to detect faces in the input image.
2. **Face Classification**: The detected face is then passed through an InceptionResnetV1 model to classify it as real or fake.
3. **Explainability**: The application uses Grad-CAM to generate a visual explanation of the model's decision.

## Dependencies

The following libraries are required to run the application:

- `torch`
- `torchvision`
- `facenet-pytorch`
- `numpy`
- `pillow`
- `opencv-python`
- `pytorch-grad-cam`
- `gradio`

You can install these dependencies using the following command:

```sh
pip install -r requirements.txt
```

## File Structure
- app.py: The main application script.
- resnetinceptionv1_epoch_32.pth: The model checkpoint file. [Download from here](https://drive.google.com/file/d/1illqNsi4f2ziJOR7sjq5N_PgrBNrTLue/view?usp=sharing)
 or Download Via Pinata through this hash
```QmUrarTy82uk2bUzty7Rhtc2XNCJHnsJy9UQhh7sMmAj78```
- requirements.txt: List of dependencies.
- README.md: This file.

## Running the Application
To run the application locally, execute the following command:

```
python app.py
```
This will launch the Gradio interface in your default web browser. You can then upload an image and get the prediction along with the explainability visualization.

# How to Use

Go over to this Hugging Face Space:  [Deep Fake Detection](https://huggingface.co/spaces/saqib129/DeepFake-Detector)

- Upload an Image: Click on the "Upload" button and select an image file.
- Get Prediction: The app will process the image and display whether it is "real" or "fake" along with confidence scores.
- Explainability: The app will also display an image with visual explainability using Grad-CAM.

# Acknowledgments
This project uses the following open-source libraries:

- Gradio
- PyTorch
- FaceNet-PyTorch
- PyTorch-Grad-CAM

# License
This project is licensed under the MIT License.

Any Contribution is highly appreciated for this Project!
