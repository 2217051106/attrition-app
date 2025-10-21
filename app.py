import os
import pandas as pd
from flask import Flask, render_template, request
from model_util import load_model

# --- Inisialisasi Flask ---
app = Flask(__name__)

# === Load model Top 10 fitur saat server start ===
model = load_model()

# === Daftar Top 10 fitur ===
TOP_10 = [
    'OverTime_No',
    'StockOptionLevel',
    'JobRole_Manufacturing Director',
    'JobLevel',
    'EnvironmentSatisfaction',
    'JobInvolvement',
    'MaritalStatus_Single',
    'JobSatisfaction',
    'JobRole_Sales Executive',
    'BusinessTravel_Travel_Frequently'
]

# === ROUTES ===
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard_view.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    # Inisialisasi variabel agar aman
    result = None
    prob = None

    if request.method == 'POST':
        try:
            # --- Inisialisasi semua fitur dengan nilai 0 ---
            data = {col: [0] for col in TOP_10}

            # --- Ambil input dari form HTML ---
            overtime = request.form['OverTime']
            data['OverTime_No'] = [1 if overtime == 'No' else 0]

            data['StockOptionLevel'] = [float(request.form['StockOptionLevel'])]
            data['JobLevel'] = [float(request.form['JobLevel'])]
            data['EnvironmentSatisfaction'] = [float(request.form['EnvironmentSatisfaction'])]
            data['JobInvolvement'] = [float(request.form['JobInvolvement'])]

            marital = request.form['MaritalStatus']
            data['MaritalStatus_Single'] = [1 if marital == 'Single' else 0]

            data['JobSatisfaction'] = [float(request.form['JobSatisfaction'])]

            job_role = request.form['JobRole']
            data['JobRole_Manufacturing Director'] = [0 if job_role == 'Manufacturing Director' else 1]
            data['JobRole_Sales Executive'] = [1 if job_role == 'Sales Executive' else 0]

            travel = request.form['BusinessTravel']
            data['BusinessTravel_Travel_Frequently'] = [1 if travel == 'Travel_Frequently' else 0]

            # --- Buat DataFrame input ---
            df = pd.DataFrame(data)

            # --- Prediksi probabilitas Attrition Yes ---
            prob_value = model.predict_proba(df)[0][1]
            prob = f"{prob_value * 100:.2f}"  # ubah ke persen 2 desimal

            # --- Prediksi label berdasarkan threshold 0.5 ---
            pred = 1 if prob_value >= 0.6 else 0

            # --- Interpretasi hasil ---
            if pred == 1:
                result = "🚨Attrition : Yes atau Karyawan kemungkinan BERHENTI"
            else:
                result = "✅ Attrition : No atau Karyawan kemungkinan TIDAK BERHENTI"

        except Exception as e:
            result = f"❌ Terjadi kesalahan: {e}"

    # --- Render template (selalu dieksekusi) ---
    return render_template(
        'predict_view.html',
        result=result,
        prob=prob,
        form=request.form if request.method == 'POST' else None
    )


# === MAIN ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Flask server berjalan di http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
