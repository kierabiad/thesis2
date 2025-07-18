import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('/content/drive/MyDrive/ThesisBacterialBlightDataset/trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Bacterial Blight Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("CASSAVA PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "/content/drive/MyDrive/ThesisBacterialBlightDataset/home_page.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Bacterial Blight Disease Recognition System! 🌿🔍
    
    Our goal is to assist in the swift and accurate identification of plant diseases. Simply upload an image of a plant, and our system will analyze it for any disease symptoms. Let's work together to safeguard crops and promote healthier harvests!
    """)

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    The dataset for this study was collected directly from cassava farmers’ plantations in Brgy. San Antonio, Kalayaan, Laguna, and nearby farm areas. It comprises images of both healthy crops and those affected by blight diseases, ensuring real-world agricultural relevance. By specifically focusing on blight-infected crops, the dataset includes images of various blight diseases alongside their healthy counterparts for comparison.
Collaboration with local farmers and agricultural experts helps capture the unique challenges faced in the region. The emphasis on blight-related data makes the dataset highly relevant to improving crop disease detection for health monitoring. Blight is a major concern for farmers due to its rapid spread and potential for significant losses, making this study’s focus both practical and necessary.
The dataset consists of 794 RGB images categorized into eight different classes, with a training-to-validation split of 77.95% and 22.05%, respectively, while preserving the original directory structure. Additionally, a separate directory containing 12 test images was later created for prediction purposes.
    
    
    Content
    1. Train (615 images)
    2. Valid (174 image)
    3. Test (12 images)
    """)


elif(app_mode=="Bacterial Blight Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    # Display image when uploaded (not dependent on a button)
    if test_image is not None:
        st.image(test_image, use_container_width=True)
    
    #Prediction Button
    if(st.button("Predict")) and test_image is not None:
        with st.spinner('Wait for it...'):
            st.write("The Plant is: ")
            result_index = model_prediction(test_image)
            #Define Class
            class_name = ['Baging_Healthy','Baging_Mild','Baging_Moderate','Baging_Severe',
                         'Cassava_Healthy','Cassava_Mild','Cassava_Moderate','Cassava_Severe']
            
            prediction = class_name[result_index]
            st.success("Model is Predicting it's a {}".format(prediction))
            
            # Add severity warnings based on prediction
            if 'Mild' in prediction:
                st.warning("⚠️ Mild infection detected. Early stage bacterial blight may be present. Recommended actions: Monitor the plant closely and consider applying copper-based bactericides as a preventive measure.")
            
            elif 'Moderate' in prediction:
                st.warning("⚠️⚠️ Moderate infection detected. Bacterial blight is established and spreading. Recommended actions: Remove infected leaves, improve air circulation, apply appropriate bactericides, and isolate the plant from healthy ones.")
            
            elif 'Severe' in prediction:
                st.error("⚠️⚠️⚠️ Severe infection detected! The plant is critically affected by bacterial blight. Recommended actions: Consider removing heavily infected plants, apply intensive treatment with appropriate bactericides, and implement strict sanitation measures to prevent spread to healthy plants.")



