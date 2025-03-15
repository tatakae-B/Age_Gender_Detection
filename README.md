# Age and Gender Detection Project

## Overview

This is a real-time age and gender detection application using OpenCV and pre-trained deep learning models. The project uses computer vision techniques to detect faces in a video stream and predict the age group and gender of detected individuals.

## Features

- Real-time face detection
- Age group estimation (8 distinct age groups)
- Gender classification (Male/Female)
- Webcam-based detection
- Simple and intuitive visualization

## Prerequisites

- Python 3.8+
- Webcam

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SudoAnirudh/Age_Gender_Detection
cd Age_Gender_Detection
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Or using conda
conda create -n Age_Gender_Detection python=3.8
conda activate Age_Gender_Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

Download the following model files from the [Age Gender Deep Learning Repository](https://github.com/GilLevi/AgeGenderDeepLearning):

- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`

Place these files in the `models/` directory.

## Project Structure

```
age-gender-detection/
│
├── main.py                 # Main application script
├── age_gender_detector.py  # Age and gender detection logic
├── requirements.txt        # Project dependencies
└── models/
    ├── age_deploy.prototxt
    ├── age_net.caffemodel
    ├── gender_deploy.prototxt
    └── gender_net.caffemodel
```

## Running the Application

```bash
python main.py
```

### Controls

- Press 'q' to quit the application

## Age Groups

The application classifies ages into the following groups:
- 0-2 years
- 4-6 years
- 8-12 years
- 15-20 years
- 25-32 years
- 38-43 years
- 48-53 years
- 60-100 years

## Model Accuracy

Please note that the accuracy of age and gender predictions depends on:
- Lighting conditions
- Face angle
- Image quality
- Model limitations

## Troubleshooting

- Ensure your webcam is connected and working
- Check that all required dependencies are installed
- Verify model files are correctly downloaded and placed in the `models/` directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here, e.g., MIT License]

## Disclaimer

This project is for educational and research purposes. The age and gender predictions are estimations and may not always be 100% accurate.
