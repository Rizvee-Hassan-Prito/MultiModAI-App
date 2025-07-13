import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
import os
import requests
import uuid
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


st.title("MultiModAI 🤖🖼️🎞️🔊📊")
user_input = st.text_input("Enter Your Name:")

# Store session state for controlling flow
if "responded" not in st.session_state:
    st.session_state["responded"] = False

# Respond button
if st.button("Submit"):
    if user_input:
        st.session_state["responded"] = True
    else:
        st.warning("Please enter your name first.")

# After clicking "Respond"
if st.session_state["responded"]:
    st.header(f"Hello, {user_input} 👋")

    st.subheader("🖼️ Image Uploader")

    # File uploader widget
    uploaded_file = st.file_uploader("**Choose an image:**", type=["jpg", "jpeg", "png"])
    
    width = st.number_input("Enter width (px):", min_value=100, value=300)
    height = st.number_input("Enter height (px):", min_value=100, value=300)

    if st.button("Upload",key="Image"):
        if uploaded_file is not None:
            # Open image using PIL
            image = Image.open(uploaded_file)

            # Show image
            resized_image = image.resize((width, height))
            st.image(resized_image, caption="Uploaded Image", use_container_width=False)
            st.success("✅ Image uploaded successfully!")

            unique_id = uuid.uuid4()
            output_directory = os.path.dirname(os.path.abspath(__file__))+'/uploads/'
            output_filename  = f"{unique_id}.jpg"
            full_path = os.path.join(output_directory, output_filename)
            image.save(full_path)

            with st.spinner('🌀 Analyzing image, please wait...'):
                response = requests.get(f"http://127.0.0.1/img_to_txt", params={"img":full_path})
            reply= response.json()['txt']
            st.markdown(f"<span style='font-size:24px; color: lightgreen; font-weight:bold;'>Caption: {reply}</span> ", unsafe_allow_html=True)


    st.subheader ("🎥 Video Uploader")

    uploaded_file = st.file_uploader("**Choose a video:**", type=["mp4", "avi", "mov"])

    if st.button("Upload",key="Video"):
        if uploaded_file is not None:
            
            unique_id = uuid.uuid4()
            output_directory = os.path.dirname(os.path.abspath(__file__))+'/uploads/'
            video_path = os.path.join(output_directory, str(unique_id)+'_'+uploaded_file.name)
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Show video
            st.video(video_path)
            st.success("✅ Video uploaded successfully!")
            
            with st.spinner('🌀 Analyzing the video, please wait...'):
                response = requests.get(f"http://127.0.0.1/vid_to_txt", params={"vid":video_path})
            reply= response.json()['txt']
            st.markdown(f"<span style='font-size:24px; color: lightgreen; font-weight:bold;'>Predicted Action: {reply}</span> ", unsafe_allow_html=True)


    st.subheader ("🔊 Audio Uploader")
    uploaded_audio_file = st.file_uploader("**Choose an audio:**", type=["m4a","mp3"])
    
    if st.button("Upload", key="Audio"):
        if uploaded_audio_file is not None:
            
            unique_id = uuid.uuid4()
            output_directory = os.path.dirname(os.path.abspath(__file__))+'/uploads/'
            audio_path = os.path.join(output_directory, str(unique_id)+'_'+uploaded_audio_file.name)

            # audio_path = os.path.join('./', uploaded_audio_file.name)
            
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio_file.getbuffer())

            # Show audio
            st.success("✅ Audio uploaded successfully!")
            st.audio(audio_path)
           
            with st.spinner('🌀 Transcribing the audio, please wait...'):
                response = requests.get(f"http://127.0.0.1/aud_to_txt", params={"aud":audio_path})
            reply= response.json()['txt']
            st.markdown(f"<span style='font-size:24px; color: lightgreen; font-weight:bold;'>Transcribed text:<br>{reply}</span> ", unsafe_allow_html=True)
            

if st.session_state["responded"]:
    st.sidebar.title("Chatbot 🤖")

    with st.sidebar:
        
        messages = st.container(height=300)
        prompt = st.chat_input(
            "Say something and/or attach an image",
            accept_file=True,
            file_type=["jpg", "jpeg", "png"],
        )

        if prompt:
            if prompt['files']:
                messages.chat_message("user").image(prompt['files'][0])
                messages.chat_message("assistant").image(prompt['files'][0])
            else:
                messages.chat_message("user").write(prompt.text)
                response = requests.get(f"http://127.0.0.1/prompt", params={"prompt":prompt.text})
                reply= response.json()['reply']
                #messages.chat_message("assistant").write(hugf_Model(prompt.text))
                messages.chat_message("assistant").write(reply)
                    

if st.session_state["responded"]:
    st.subheader("📊 Feature-Based Plotting")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        if "df" not in st.session_state:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df 
            st.success("Dataset uploaded successfully!")
        
        df= st.session_state.df
        st.write("### Dataset Preview:")
        st.dataframe(df.head())

        features_to_delete = st.selectbox(f"**Select a feature (column) to delete**", df.columns) 
        if st.button("Delete",key="feature",icon="🚨"):
            df=df.drop(columns=features_to_delete)
            st.session_state.df = df 
            st.success(f"Selected Feature: {features_to_delete} is deleted")

        # Column selector
        feature = st.selectbox("Select a feature (column) to plot", df.columns)

        if feature:
            # Detect feature type
            if pd.api.types.is_numeric_dtype(df[feature]):
                st.write(f"### Selected Feature: {feature} (Numerical)")

                plot_type = st.radio("Choose a plot type", ["Histogram", "Boxplot"])

                fig, ax = plt.subplots()
                if plot_type == "Histogram":
                    sns.histplot(df[feature], kde=True, ax=ax)
                else:
                    sns.boxplot(x=df[feature], ax=ax)

            else:
                st.write(f"### Selected Feature: {feature} (Categorical)")
                fig, ax = plt.subplots()
                sns.countplot(x=df[feature], ax=ax)

            st.pyplot(fig)

        st.subheader("📈 Train, Test and Predict Category")
        target = st.selectbox("**Select target column**", df.columns)

        if pd.api.types.is_numeric_dtype(df[target]):
            st.write(f"###  {feature} is not a categorical column. Select appropriate column.")
        
        if not pd.api.types.is_numeric_dtype(df[target]):
            st.write(" **Select data points for prediction:**")


            st.session_state.input = df.drop(columns=[target])
            x = st.session_state.input
            user_input = {}

            for col in x.columns:
                if x[col].dtype == 'object':
                    options = x[col].unique().tolist()
                    user_input[col] = st.selectbox(f"{col}", options=options)
                else:
                    user_input[col] = st.number_input(f"{col}", value=float(x[col].mean()))


            if st.button("Train, Test and Predict",key="target"):

                # Save to session state if you want to use them later
                st.session_state.X = df.drop(columns=[target])
                st.session_state.y = y= df[target]

                def addressing_outliers(col):
                    q1=st.session_state.X[col].quantile(0.25)
                    q3=st.session_state.X[col].quantile(0.75)
                    iqr=q3-q1
                    lower_bound=q1-1.5*iqr
                    upper_bound=q3+1.5*iqr
                    st.session_state.X[col] = st.session_state.X[col].clip(lower=lower_bound, 
                                                                        upper=upper_bound)


                le = LabelEncoder()
                for col in st.session_state.X.select_dtypes(include=["object"]).columns:
                    st.session_state.X[col] = le.fit_transform(st.session_state.X[col])
                
                
                for col in st.session_state.X.columns:
                    addressing_outliers(col)

                # Normalize features
                scaler = StandardScaler()
                x= scaler.fit_transform(st.session_state.X)
                #x = st.session_state.X

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

                rf_model = RandomForestClassifier(n_estimators=500, min_samples_split=4)
                
                
                rf_model.fit(x_train, y_train)
                y_pred = rf_model.predict(x_test)
                

                y_pred = rf_model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f'Accuracy: {accuracy:.4f}')
                st.write('Classification Report:')
                st.text(classification_report(y_test, y_pred))

                with st.spinner('🌀 Model evaluation is in progress. Please wait...'):
                    skf = StratifiedKFold(n_splits=10, shuffle=True)
                    cv_scores = cross_val_score(rf_model, x, y, cv=skf, scoring='accuracy')

                    st.write(f'Cross-Validation Accuracy: {np.mean(cv_scores):.4f}')

                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                # Plot using ConfusionMatrixDisplay
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
                fig, ax = plt.subplots(figsize=(6, 4))
                disp.plot(ax=ax, cmap='Blues')

                # Display in Streamlit
                st.pyplot(fig)

                # Prepare input DataFrame
                input_df = pd.DataFrame([user_input])

                # Encode input using same encoders
                for col in input_df.select_dtypes(include='object').columns:
                    input_df[col] = le.transform(input_df[col])

                prediction = rf_model.predict(input_df)[0]
                st.markdown(
                f"<h3 style='color: lightgreen;'>✅ Predicted class: {prediction}</h3>",
                unsafe_allow_html=True
                )

