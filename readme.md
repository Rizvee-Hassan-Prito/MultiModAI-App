# MultiModAI: A Streamlit-Based Multi-Modal Intelligence and CSV Analysis Web App 🤖📊🖼️🎞️🔊

A Streamlit and FastAPI powered application that unifies multiple pretrained large language models (LLMs) from HuggingFace Models to provide a seamless user experience for multimodal AI tasks—text interaction, image captioning, video action recognition, audio transcription—and interactive CSV dataset classification using a Random Forest classifier.

---

## 🔍 Overview
MultiModAI app is an all-in-one intelligent assistant application that allows users to interact with various forms of data—text, image, audio, video, and structured tabular data. Built using Streamlit for the frontend and FastAPI as the backend, this app connects several pre-trained models from the HuggingFace Model Zoo to provide cutting-edge AI functionalities.

It begins by capturing the user's name, then presents a clean UI for:

- Conversational AI with a pre-trained LLM model.

- Classifying actions from videos.

- Captioning JPEG/PNG images.

- Transcribing MP3 audio files.

- Performing data analysis and classification on CSV datasets

Whether you're a data scientist, educator, student, or AI enthusiast, this app provides a simple way to explore AI capabilities across different data types without writing code.

---

## 📸 UI Image  
![0_1](https://github.com/user-attachments/assets/1dbcfb90-86a0-4b2a-9206-23abc781fcd6)
![0_2](https://github.com/user-attachments/assets/337f91e4-acdc-42a3-9e05-9e9cd42e0c04)
![1](https://github.com/user-attachments/assets/f07a88cb-39db-497f-9c42-5a194c416444)
![2](https://github.com/user-attachments/assets/6c7f6d63-5f46-430e-8723-269697939697)
![3](https://github.com/user-attachments/assets/724138f8-dc92-41cc-be9d-369101fc2757)
![4](https://github.com/user-attachments/assets/cd68cdd5-f201-4123-9ff0-616cef6344f1)
![5](https://github.com/user-attachments/assets/843ac5c3-3ae7-4f38-8793-16d806835d37)
![6](https://github.com/user-attachments/assets/770e37d4-3574-4ed0-85fd-eae272800d2b)
![7](https://github.com/user-attachments/assets/4d628182-bb4e-4ff4-957d-25d34833d3dc)
![8_0](https://github.com/user-attachments/assets/d8d15c73-7999-411a-8348-0325182630cb)
![8_1](https://github.com/user-attachments/assets/3d79a724-4b6c-43ad-b328-15e8ad28d813)
![9](https://github.com/user-attachments/assets/bc20802f-2ce1-4291-a1ca-3be535dd2142)
![10](https://github.com/user-attachments/assets/5d425261-764f-478b-8483-b1e0c132179f)


## 🚀 Features

### 👤 User Initialization

- Users must input their name before accessing app features, creating a personalized session.

### 💬 Chatbot (Qwen/QwQ-32B)

- Chat with a powerful LLM.

- Handles general-purpose conversation.

### 🖼️ Image to Text (Caption Generation)

- Model: Salesforce/blip-image-captioning-base

- Accepts JPEG/PNG and generates descriptive captions.

### 🎞️ Video to Text (Action Recognition)

- Model: facebook/timesformer-base-finetuned-k600

- Upload MP4 video to recognize and classify the main action in the video.

### 🔊 Audio to Text (Transcription)

- Model: openai/whisper-tiny

- Upload MP3 audio and receive transcriptions of a few sentences.

### 📊 CSV Dataset Analysis

#### 📈 Data Visualization
- Create Histogram, Box Plot, and Bar Plot for feature-level insights:
        
    - Histograms/Box Plots for numerical features.

    - Bar Plots for categorical features.

- Users can select the feature to visualize from a dropdown.

#### 🧹 Data Preprocessing
- Remove unwanted columns interactively.

- Automatically removes outliers from numerical features.

- Automatically standardizes nuemerical features before training.

#### 🌲 Categorical Prediction
- Model: Random Forest Classifier

- Users can select target variable via UI.

- Training occurs with: 75/25 Train-Test Split

- Displays:

    - Prediction result

    - Test set accuracy

    - 10-fold cross-validation score
    
---

## ▶️ How to Run
1. Clone the Repository

```bash
git clone https://github.com/NxtVis/vision-prito/MultiModAI App_Streamlit_FastAPI.git
cd MultiModAI App_Streamlit_FastAPI
```
2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```
3. Install Required Packages
```bash
pip install -r requirements.txt
```
4. Start the FastAPI Backend
```bash
cd server
python fstapi.py
```
5. Launch the Streamlit App
```bash
streamlit run app.py
```
---
## 🚧 Scope and Limitations

### ✅ Scope:
- Unified platform for multi-modal AI (text, image, audio, video).

- Light-weight interactive ML modeling using tabular data.

- Data preprocessing and visualization built-in.

- Educational and prototyping use cases.

### ⚠️ Limitations:
- Only supports:
    - .jpg, .jpeg, .png for image input

    - .mp4 for video

    - .mp3 for audio

- Chatbot does not support image inputs.

- Performance may vary based on system resources (RAM/CPU/GPU).

- No persistent user session or dataset storage.

- LLM/chatbot may have limitations on factual reliability or image-based understanding.

## 🎯 Future Targets

- 🔄 Add support for more ML models (e.g., logistic regression, AdaBoost, SVM). Include regression and clustering ML models.

- 📊 Incorporate feature importance and SHAP (SHapley Additive exPlanations) for interpretability.

- 🧠 Fine-tune models for domain-specific use cases.

- 💾 Add database support for session and data persistence.

- ☁️ Enable cloud or Docker deployment for scalable usage.

- 🔐 Add authentication and access control for multiple users.
