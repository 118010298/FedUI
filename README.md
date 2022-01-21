# FedUI
A code-free user interface for industry level federated learning object detection tasks
1. Description
Invited by Guangdong Power Grid Co., LTD., we examined a federal learning system for object detection tasks called FedVision, and built corresponding user interface to facilitate the use of the system by some staff in power grid companies who are not computer professional background. The GUI is divided into four main modules, which are dataset configuration module, model parameter configuration module, online visualization of training results module, and model evaluation and release module.

1.1 Dataset configuration module 
• Design Concept: This module can read the image format data set including VOC and COCO, and read the basic information of the data set (image number, annotation type and the number of each annotation), and visualize this information in the form of statistical charts. The module also supports data set segmentation, can actively modify the proportion of training set, test set and verification set, and respectively save in the corresponding path.
• Development progress: At present, this module has implemented all the functions mentioned in the Design Concept, but only for VOC and COCO format datasets, and will be developed for other formats in the future.

1.2 Model parameter configuration module
• Design Concept: This module can configure the parameters of federated learning object detection task. Parameter setting is mainly divided into two parts. One is the parameter setting related to the federated learning global model, such as the maximum trainer, the longest waiting time and whether to use GPU training, etc.; the other part is the setting of edge model, such as Backbone selection, image input size and batch size. It is worth noting that the FedVision framework uses configuration files in .yaml format to configure parameters and start tasks on each participant's machine by transferring and reading .yaml files. Therefore, this module of the GUI can read and modify the relevant .yaml configuration files to configure parameters and start training environments and tasks.
• Development progress: The parameter configuration module has been comprehensively developed for FedVision's existing model based on FedAvg, including tasks of edge model Backbone for CSPDarknet, Mobilenet and Resnet. In future development, we hope to build more types of models ourselves through built-in methods in the Paddle Detection library and support them in the GUI.

1.3 Online visualization of training results module
• Design Concept: This module can visualize training results. This module can read the task log file of local participants, and through the visualization function of Bokeh library, display the training indicators including loss and MAP of local models in real time at a certain refresh frequency, and draw relevant statistical charts.
• Development progress: At present, this module can only directly visualize loss indicators of training by reading log files, while indicators such as BboxMAP need to be evaluated and visualized by storing models and turning them into the framework of Paddle because FedVision does not expose any relevant interfaces at present. That is, the Loss index can be updated online in real time, while other indexes such as MAP cannot be displayed in the log, so they can only be displayed after the training is completed. In the future work, we hope to modify the FedVision source code to expose part of the measurements API to expand the training KPI that can be visualized.

1.4 Model evaluation and release module
• Design Concept: This module aims to evaluate model indicators and publish models. The model evaluation and publishing module can summarize the model training index, generate model calculation diagram, export and save the training model, and provide online uploading and other functions.
• Development progress: At present, except the function of exporting and saving training model, other functions need to be supported by the Paddle library other than FedVision. In the future development, we will try to transplant the functions of this part of Paddle to FedVision framework.


2. Video Link
The below link will direct to a video I recorded to demonstrate the workflow of the Alpha test version of our framework. I'm sorry to inform you that due to the needs of Guangdong Power Grid Co., LTD., the UI is only in simplified Chinese at present, which may cause you to be unable to intuitively understand the meaning of some keys.
https://drive.google.com/file/d/1iNeelzvALAEQ39KsJtKtCvJ8W6o46Y2v/view?usp=sharing
