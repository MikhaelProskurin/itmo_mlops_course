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

This project delivers a service for automatically identifying aircraft in images captured from aerial or space-based sources. It streamlines the development and deployment of object detection models, offering experiment tracking and scalable serving capabilities to provide valuable insights from imagery data.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Contributing](#contributing)
- [Citation](#citation)

---
## Core features

1. **Object Detection Service**: The core functionality of the project is a service that detects objects (specifically aircraft) within images, utilizing computer vision techniques.
2. **BentoML Deployment**: The model is packaged and served using BentoML, enabling easy deployment and scalability as a REST API. The bentofile.yaml configures the service.
3. **ClearML Integration**: The project integrates with ClearML for experiment tracking, logging metrics, artifacts (models, datasets), and managing the ML lifecycle. This includes configuration via clearml-init.
4. **Model Training Pipeline**: The `main.py` script provides a pipeline for training, tuning, or evaluating the object detection model. It leverages PyTorch and potentially ONNX conversion.
5. **Dataset Handling**: The project utilizes a dataset of military aircraft images (from Kaggle) and defines specific directory structures for annotations (XML, JPG, TXT). Preprocessing steps are included to prepare the data.
6. **Experiment Management**: The `experiment.py` module encapsulates the training and evaluation logic, providing a structured way to manage experiments with configurable parameters and logging capabilities.
7. **Data Preprocessing**: Preprocessing scripts (`preprocessing.py`) parse XML annotation files and prepare the dataset for model training, including extracting bounding box coordinates and labels.

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
