# itmo_mlops_course

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![aiohttp](https://img.shields.io/badge/AIOHTTP-2C5BB4.svg?style={0}&logo=AIOHTTP&logoColor=white)
![bentoml](https://img.shields.io/badge/BentoML-000000.svg?style={0}&logo=BentoML&logoColor=white)
![jinja2](https://img.shields.io/badge/Jinja-B41717.svg?style={0}&logo=Jinja&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![onnx](https://img.shields.io/badge/ONNX-005CED.svg?style={0}&logo=ONNX&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![pydantic](https://img.shields.io/badge/Pydantic-E92063.svg?style={0}&logo=Pydantic&logoColor=white)

---

## Overview

This project delivers a service for automatically identifying aircraft in images captured from aerial or space-based sources. It streamlines the process of object detection, offering a scalable solution with experiment tracking and model management capabilities to ensure reliable performance.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

---
## Core features

1. **Object Detection Service**: The core functionality of the project is a service that detects objects (specifically aircraft) within images, utilizing computer vision techniques.
2. **BentoML Integration**: The model is deployed as a BentoML service, enabling easy serving and scalability. This allows for creating a REST API endpoint for predictions.
3. **ClearML Experiment Tracking**: Integration with ClearML provides experiment tracking, logging of metrics, artifacts (models, datasets), and visualization capabilities during training and evaluation.
4. **Model Training Pipeline**: The project includes a pipeline for training the object detection model, likely using PyTorch based on the code snippets. This involves data loading, preprocessing, and optimization.
5. **Dataset Handling**: The project handles a specific dataset of military aircraft images (from Kaggle) and defines a directory structure for annotations and image data. It also supports different sample sizes for training/evaluation.
6. **Data Preprocessing**: Preprocessing steps are implemented to prepare images and annotations for training, including parsing XML annotation files and creating a suitable dataset format.
7. **Bounding Box Prediction & Visualization**: The service predicts bounding boxes around detected aircraft in the input images and visualizes these predictions with confidence scores.

---

## Installation

Install itmo_mlops_course using one of the following methods:

**Build from source:**

1. Clone the itmo_mlops_course repository:
```sh
git clone https://github.com/MikhaelProskurin/itmo_mlops_course
```

2. Navigate to the project directory:
```sh
cd itmo_mlops_course
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Documentation

A detailed itmo_mlops_course description is available [here](https://github.com/MikhaelProskurin/itmo_mlops_course/tree/master/mkdocs_temp/docs).

---

## Contributing

- **[Report Issues](https://github.com/MikhaelProskurin/itmo_mlops_course/issues)**: Submit bugs found or log feature requests for the project.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    MikhaelProskurin (2024). itmo_mlops_course repository [Computer software]. https://github.com/MikhaelProskurin/itmo_mlops_course

### BibTeX format:

    @misc{itmo_mlops_course,

        author = {MikhaelProskurin},

        title = {itmo_mlops_course repository},

        year = {2024},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/MikhaelProskurin/itmo_mlops_course.git}},

        url = {https://github.com/MikhaelProskurin/itmo_mlops_course.git}

    }

---
