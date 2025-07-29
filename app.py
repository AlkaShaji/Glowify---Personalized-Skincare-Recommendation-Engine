from flask import Flask, request, render_template, jsonify, Response, redirect, url_for
import cv2
import numpy as np
import pandas as pd 
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from reco import SkinCareRecommender
import os
import io
import time
import inspect



app = Flask(__name__)


# Path configuration for uploading images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
skin_type_model_path = os.path.join(BASE_DIR, r'C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\models\efficientnetb0_skin_type_model.keras')
skin_concern_model_path =os.path.join(BASE_DIR, r'C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\models\resnet_skin_concern_finetuned.keras')

skin_type_model = load_model(skin_type_model_path)
skin_concern_model = load_model(skin_concern_model_path)

# Define the CSV file path
CSV_FILE_PATH = r"C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\Skinpro_with_ratings.csv"

# Load product details (assuming you need this for the recommender system)
df = pd.read_csv(CSV_FILE_PATH)


# Load face detection model using DNN
face_net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, 'models/face_detector/deploy.prototxt'), 
    os.path.join(BASE_DIR, 'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
)

# Make sure this matches your class label order
skin_type_labels = ['combination', 'dry', 'normal', 'oily', 'sensitive']
skin_concern_labels = ['acne', 'dark circles', 'eyebags', 'pigmentation', 'redness', 'scar', 'wrinkle']


camera = cv2.VideoCapture(0)


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# === Helper Functions ===
def detect_face(frame):
    
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            face = frame[y:y1, x:x1]
            return face
    return None

def preprocess_image(image, target_size=(224, 224)):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image = cv2.resize(image, target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)


def get_skin_concern_labels(preds, threshold=0.2):
    return [skin_concern_labels[i] for i, val in enumerate(preds) if val >= threshold]

#Accesses your webcam in real-time using OpenCV,
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break 

        # Face detection and prediction here 
        face_roi = detect_face(frame)
        predicted_skin_type = "no face"
        predicted_concerns = []

        if face_roi is not None:
            predicted_skin_type = predict_skin_type(face_roi)
            predicted_concerns = predict_skin_concern(face_roi)
            
        cv2.putText(frame, f"Skin Type: {predicted_skin_type}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Concerns: {', '.join(predicted_concerns)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

      
        # Only now encode for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
                
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

   
def predict_skin_type(face_roi):
    try:
        processed_face = preprocess_image(face_roi)
        skin_type_pred = skin_type_model.predict(processed_face)[0]
        return skin_type_labels[np.argmax(skin_type_pred)]
    except Exception as e:
        print(f"Error in predict_skin_type: {e}")
        return "Error"

def predict_skin_concern(face_roi):
    try:
        processed_concern = preprocess_image(face_roi)
        concern_preds = skin_concern_model.predict(processed_concern)[0]
        return get_skin_concern_labels(concern_preds, threshold=0.2)
    except Exception as e:
        print(f"Error in predict_skin_concern: {e}")
        return ["Error"]
       
def get_recommendations(skin_type, concerns, df, top_n=5):
    try:
        df.columns = df.columns.str.strip()  # Just to be sure
        print(f"Columns in the DataFrame: {df.columns}")

        # Ensure the 'Concern' column exists
        if 'Concern' not in df.columns:
            print("Error: 'Concern' column not found in the DataFrame!")
            return pd.DataFrame()  # Return an empty DataFrame in case of error

        recommended_products = df[
            (df['Skintype'].str.lower() == skin_type.lower()) &
            (df['Concern'].str.lower().isin([c.lower() for c in concerns]))
        ].drop_duplicates(subset=['Product']).head(top_n)

        return recommended_products
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        return pd.DataFrame()   # Return an empty DataFrame in case of an error

# Prepare product details
product_details = pd.DataFrame({
    'product_id': range(1, len(df) + 1),
    'skin_type': df['Skintype'],
    'concerns': df['Concern']
})

# Encode
skin_encoder = LabelEncoder()
concern_encoder = LabelEncoder()

product_details['skin_type_encoded'] = skin_encoder.fit_transform(product_details['skin_type'])
product_details['concerns_encoded'] = concern_encoder.fit_transform(product_details['concerns'])

product_features = product_details[['skin_type_encoded', 'concerns_encoded']]
product_similarity = cosine_similarity(product_features)

product_similarity_df = pd.DataFrame(product_similarity,
                                     index=product_details['product_id'],
                                     columns=product_details['product_id'])

df['product_id'] = product_details['product_id']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    ret, frame = camera.read()
    if not ret:
        return "Failed to capture image", 500

    face = detect_face(frame)
    if face is None:
        return render_template('results.html', skin_type="No Face", skin_concerns=["Face not detected"], products=[])

    # Predict
    skin_type = predict_skin_type(face)
    skin_concerns = predict_skin_concern(face)

    df = pd.read_csv(CSV_FILE_PATH)
    recommended_products = get_recommendations(skin_type, skin_concerns, df)

    return render_template('results.html', 
                           skin_type=skin_type, 
                           skin_concerns=skin_concerns, 
                           products=recommended_products)



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    image_file = request.files['image']
    if image_file.filename == '' or not allowed_file(image_file.filename):
        return redirect(request.url)
    

     # Define and save the uploaded image
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)

    image_url = url_for('static', filename=f'uploads/{filename}')
    img = cv2.imread(filepath)
    
    try:
        face = detect_face(img)
        if face is None:
            return render_template('results.html', skin_type="No Face", skin_concerns=["Face not detected"], products=[], image_url=image_url)
        
        face_for_type = preprocess_image(face, (224, 224))
        face_for_concern = preprocess_image(face, (224, 224))

        skin_type_pred = skin_type_model.predict(face_for_type)[0]
        skin_concern_pred = skin_concern_model.predict(face_for_concern)[0]

        skin_type = skin_type_labels[np.argmax(skin_type_pred)]
        skin_concerns = get_skin_concern_labels(skin_concern_pred, threshold=0.2)

        # Debugging: Print out the concerns with their predicted values
        print(f"Skin Concerns and their predicted values: {dict(zip(skin_concern_labels, skin_concern_pred))}")

        df = pd.read_csv(CSV_FILE_PATH)
        recommended_products = get_recommendations(skin_type, skin_concerns, df)

    except Exception as e:
        return render_template('results.html', skin_type="Error", skin_concerns=[str(e)], products=[], image_url=image_url)

    return render_template('results.html', 
                           skin_type=skin_type, 
                           skin_concerns=skin_concerns, 
                           recommended_products=recommended_products,
                           image_url=image_url)

@app.route('/recommend')
def recommend_page():
    return render_template('recommendation.html')

recommender = SkinCareRecommender(r'C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\Skinpro_with_ratings.csv')

@app.route('/recommend_products', methods=['POST'])
def recommend_products():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        print(f"Received data: {data}")

        skin_type = data.get('skin_type', '').strip()
        concerns = data.get('skin_concerns', '').strip()

        if not skin_type or not concerns:
            return jsonify({"error": "Missing skin_type or skin_concerns"}), 400

        if isinstance(concerns, str):
            concerns = [concerns]  # In case concerns come as a string instead of a list

       
        recommended_products = recommender.get_recommended_products(skin_type, concerns)

        if recommended_products.empty:
            return jsonify({"message": "No products found"}), 200

        # Now render the recommendation page with the product recommendations
        return render_template('recommendation.html', recommendations=recommended_products.to_dict(orient='records'))
    
    except Exception as e:
        print(f"Error in /recommend_products: {e}")
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(debug=True)