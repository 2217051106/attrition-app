import mlflow
import os
import joblib

def get_model():
    """
    Download model XGBoost Top 10 fitur dari DagsHub (MLflow Model Registry)
    dan simpan ke lokal untuk digunakan oleh Flask.
    """
    # 🔗 Ganti dengan URI tracking proyek kamu
    uri_artifacts = "https://dagshub.com/2217051106/attrition-model.mlflow"
    mlflow.set_tracking_uri(uri_artifacts)

    # 📝 Nama model di registry dan versinya
    model_uri = "models:/xgb_attrition_model_top10/1"   # ganti 1 sesuai versi model kamu

    # Load model langsung dari DagsHub (pakai flavor xgboost)
    model = mlflow.xgboost.load_model(model_uri)

    # Simpan lokal biar Flask gak perlu request ke internet terus
    if not os.path.exists("model"):
        os.makedirs("model")
    local_path = "model/xgb_attrition_model_top10_v1.pkl"
    joblib.dump(model, local_path)

    print(f"✅ Model Top 10 fitur berhasil diunduh dari DagsHub dan disimpan ke: {local_path}")
    return model


def load_model():
    """
    Load model Top 10 fitur dari file lokal (jika sudah pernah diunduh).
    Kalau belum ada, otomatis unduh dari DagsHub.
    """
    local_path = "model/xgb_attrition_model_top10_v1.pkl"

    if os.path.exists(local_path):
        model = joblib.load(local_path)
        print(f"✅ Model lokal dimuat dari: {local_path}")
        return model
    else:
        print("📥 Model lokal belum ada — mengunduh dari DagsHub...")
        return get_model()
