import joblib
from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Loading model
Model = joblib.load('model_pipeline.pkl')

# Dataset
try:
    df_data = pd.read_csv("dataset.csv")
    df_data.columns = df_data.columns.str.strip()
except:
    print("Unable to load dataset.")

# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction process : Predict with data taken from user
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # form data
        inputs = {
            # --- Sayısal Değerler (Float veya Int'e çeviriyoruz) ---
            'Net Metrekare': float(request.form['area']),  # HTML name="area"
            'Brüt Metrekare': float(request.form['bedrooms']),  # HTML name="bedrooms" ama Label "Brüt m2"
            'Aidat': float(request.form['bathrooms']),  # HTML name="bathrooms" ama Label "Aidat"
            'Binanın Kat Sayısı': int(request.form['stories']),  # HTML name="stories"

            # --- Kategorik Değerler (Olduğu gibi String alıyoruz) ---
            'Oda Sayısı': request.form['oda_sayisi'],
            'Balkon Durumu': request.form['mainroad'],  # HTML name="mainroad" ama Label "Balkon"
            'Banyo Sayısı': request.form['guestroom'],  # HTML name="guestroom" ama Label "Banyo"
            'Isıtma Tipi': request.form['hotwaterheating'],
            'Site İçerisinde': request.form['airconditioning'],
            'Bulunduğu Kat': request.form['bulundugu_kat'],
            'Binanın Yaşı': request.form['binanin_yasi'],
            'Tipi': request.form['tipi'],
            'Kullanım Durumu': request.form['kullanim_durumu'],

            'Ilce': request.form['ilce'],
            'Mahalle': request.form['mahalle']
        }

        features = pd.DataFrame([inputs])

        prediction = Model.predict(features)[0]

        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        return f"Hata oluştu: {e}"


@app.route('/get_mahalleler/<ilce>')
def get_mahalleler(ilce):
    # 1. Veri setinden o ilçeye ait satırları filtrele
    ilce_verisi = df_data[df_data['Ilce'] == ilce]

    # 2. O ilçedeki benzersiz mahalleleri bul ve sırala
    mahalleler = sorted(ilce_verisi['Mahalle'].unique().tolist())

    # 3. JSON olarak JavaScript'e gönder
    return jsonify(mahalleler)


if __name__ == "__main__":
    app.run(debug=True)