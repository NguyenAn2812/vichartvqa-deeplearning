
# Deep Learning Final Project

This repository contains the final project submission for the **Deep Learning** course.  
The project is divided into two independent tasks, following the course requirements: a major Visual Question Answering task and an additional deep learning application task.

The course project requires students to design, implement, evaluate, and demonstrate deep learning systems, including dataset preparation, model development, experimental comparison, and final reporting. The first task focuses on Vietnamese Visual Question Answering with both custom and pretrained multimodal models, while the second task allows students to propose and implement another interesting deep learning problem. The expected submission includes source code, detailed README files, report, slides, demo video, dataset, and model checkpoints.  

## Repository Structure

```text
.
├── dataset/
├── task1/
│   ├── README.md
│   ├── Train_A1_A2.ipynb
│   ├── Train_B1.ipynb
│   ├── Train_B2.ipynb
│   ├── app.py
│
└── task2/
│   ├── README.md
│   ├── Vietnamese_Emotion_Recognition.ipynb
│
└── README.md
````
## Dataset

Folder: [`dataset`](dataset/)

## Task 1 — Vietnamese Chart Visual Question Answering

Folder: [`task1`](task1/)

Task 1 implements a Vietnamese **Visual Question Answering (VQA)** system in a specialized domain. The system receives a chart image and a Vietnamese question, then generates a Vietnamese answer.

This task follows the required experimental design of the course:

| Configuration | Description                                             |
| ------------- | ------------------------------------------------------- |
| **A1**        | Custom multimodal architecture with LSTM decoder        |
| **A2**        | Custom multimodal architecture with Transformer decoder |
| **B1**        | Zero-shot pretrained multimodal model                   |
| **B2**        | Fine-tuned pretrained multimodal model                  |

The main purpose of Task 1 is to compare two development directions:

1. **Custom multimodal architecture**
   Using separate image encoder, text encoder, fusion module, and answer decoder.

2. **Pretrained multimodal model adaptation**
   Using a pretrained vision-language model in zero-shot and fine-tuned settings.

The task includes dataset construction, training, evaluation, comparison, and a local demo interface.

## Task 2 — Additional Deep Learning Application

Folder: [`task2`](task2/)

Task 2 presents an additional deep learning application selected by the project team.
This task focuses on proposing a meaningful problem, explaining why deep learning is suitable for the task, implementing the solution, and providing a working demo.

The folder contains the full implementation, experiment results, and usage instructions for this second task.

## Main Requirements Covered

This repository is designed to satisfy the course requirements, including:

* Dataset preparation and documentation
* Model implementation and training
* Comparison between multiple model configurations
* Evaluation using suitable automatic metrics
* Experimental analysis and result visualization
* Local or web-based demo interface
* Checkpoint and dataset organization
* Final report, slides, and demo video preparation

## How to Use This Repository

Each task has its own README file with detailed instructions.

To start with Task 1:

```bash
cd Task_1_ChartVQA
```

Then follow the instructions in:

```text
Task_1_ChartVQA/README.md
```

To start with Task 2:

```bash
cd Task_2_DeepLearning_Application
```

Then follow the instructions in:

```text
Task_2_DeepLearning_Application/README.md
```

## Deliverables

The complete submission includes:

* Source code
* Dataset documentation
* Training and evaluation notebooks
* Model checkpoints
* Demo application
* Final report
* Presentation slides
* Demo video

## Notes

This repository is intended for academic use as part of the Deep Learning final project.
Each task is organized independently to make the implementation, evaluation, and demo process easier to follow.
