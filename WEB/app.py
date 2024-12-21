from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Memuat model dan vectorizer dari file pickle
model_svm = pickle.load(open('model_svm.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input teks dari pengguna
    user_input = request.form['user_input']
    print("User input received:", user_input)  # Debugging input

    # Transformasi teks menggunakan vectorizer yang dimuat
    transformed_text = vectorizer.transform([user_input])
    print("Transformed input:", transformed_text)  # Debugging transformasi

    # Prediksi menggunakan model SVM yang dimuat
    prediction = model_svm.predict(transformed_text)
    print("Prediction:", prediction)  # Debugging prediksi

    # Memastikan logika yang benar untuk menetapkan sentimen
    if prediction[0] == 1:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE"
    
    print("Sentiment:", sentiment)  # Debugging sentimen

    return render_template('index.html', user_input=user_input, sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
