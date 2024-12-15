### **Source dataset**
*  https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset

### **models directory configuration**
*  models_history/models_onnx
*  models_history/models_torch

### **annotations directory configuration**
*  src/annotations (xml files)
*  src/data (jpg files)
*  src/prepared_samples (txt files)
#### this structure (anns dir & models dir) can be replaced with an arbitrary one, but you will have to take this into account and adjust some sections of the code

### **Bentoml service**
* model which participates in service must be uploaded into bento
* run command ```bentoml serve service:<service_class_name>``` to set up a local server at ```http://localhost:3000```

### **Clearml**
* run command ```clearml-init``` for clearml.config file creation. Then pass the private keys from your clearml profile

### Experiments
* ```main.py``` responces for model **training, tuning or evaluation**
* just research and execute ```main.py``` for starting the process
* experiments would be logged on clearml
