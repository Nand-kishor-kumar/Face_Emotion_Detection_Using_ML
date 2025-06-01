from flask import Flask, render_template, Response, jsonify, url_for
import cv2
from fer import FER
import random
import atexit
import os
import numpy as np

app = Flask(__name__)
detector = FER(mtcnn=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEOS_DIR = os.path.join(BASE_DIR, 'static', 'videos')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load gender detection model
gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODELS_DIR, 'gender_deploy.prototxt'),
    os.path.join(MODELS_DIR, 'gender_net.caffemodel')
)

GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

quotes = {
    "happy": ["Keep smiling!", "Stay positive!"],
    "sad": ["It's okay to feel sad.", "Better days are coming."],
    "angry": ["Take a deep breath.", "Peace begins with a smile."],
    "surprise": ["Wow!", "Didn't expect that!"],
    "neutral": ["Keep going.", "Steady and strong."],
    "fear": ["Fear is temporary. Courage is forever."],
    "disgust": ["Let go of negativity.", "Focus on the good."]
}

emoji_dict = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "😖"
}

latest_data = {"emotion": "neutral", "gender": "Unknown"}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def release_camera():
    if cap.isOpened():
        cap.release()

atexit.register(release_camera)

def get_gender(face_img):
    if face_img.size == 0:
        return "Unknown"
    try:
        face_img_resized = cv2.resize(face_img, (227, 227))
        blob = cv2.dnn.blobFromImage(face_img_resized, scalefactor=1.0, size=(227, 227),
                                     mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        return gender
    except Exception as e:
        print("Gender detection error:", e)
        return "Unknown"

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip frame horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)

        results = detector.detect_emotions(frame)
        height, width, _ = frame.shape

        for face in results:
            (x, y, w, h) = face["box"]

            # Add padding for better face crop
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)

            face_img = frame[y1:y2, x1:x2]

            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)

            gender = get_gender(face_img)

            # Debug prints - can comment out later
            print(f"Detected Emotion: {dominant_emotion}, Gender: {gender}")

            latest_data["emotion"] = dominant_emotion
            latest_data["gender"] = gender

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{dominant_emotion.capitalize()}, {gender}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/quote')
def get_quote():
    emotion = latest_data["emotion"]
    gender = latest_data["gender"]
    quote = random.choice(quotes.get(emotion, ["Stay strong!"]))
    emoji = emoji_dict.get(emotion, "🙂")
    return jsonify({"quote": quote, "emoji": emoji, "emotion": emotion, "gender": gender})

@app.route('/play_video/<emotion>')
def play_video(emotion):
    if emotion not in quotes:
        emotion = "neutral"
   # Send the image filename, e.g. "happy.png"
    image_filename = f"{emotion}.png"
    return render_template('video_player.html', image_path=image_filename, emotion=emotion)


if __name__ == "__main__":
    app.run(debug=True)
