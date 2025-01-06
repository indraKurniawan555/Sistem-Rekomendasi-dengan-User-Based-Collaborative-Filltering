import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib

# Load dataset
def load_data(file_path):
    """
    Memuat data dari file CSV untuk pelatihan model rekomendasi
    """
    try:
        # Baca file CSV dengan delimiter yang sesuai
        df = pd.read_csv(file_path, delimiter=';')
        
        # Konversi kolom rating, mengganti koma dengan titik
        def convert_rating(value):
            # Ganti koma dengan titik, hilangkan spasi
            return float(str(value).replace(',', '.').replace(' ', ''))
        
        # Terapkan konversi pada kolom rating
        df['rating'] = df['rating'].apply(convert_rating)
        
        # Pastikan kolom yang dibutuhkan ada
        required_columns = ['user_id', 'tempat_wisata', 'rating']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Kolom {col} tidak ditemukan dalam dataset")
        
        # Tambahkan debug print
        print("Sample data setelah konversi:")
        print(df[['user_id', 'tempat_wisata', 'rating']].head())
        print("\nTipe data rating:", df['rating'].dtype)
        
        return df
    except Exception as e:
        print(f"Kesalahan membaca file: {e}")
        return None

# Persiapan data untuk Surprise
def prepare_surprise_data(df):
    """
    Mempersiapkan data untuk library Surprise
    
    Parameters:
    df (pandas.DataFrame): DataFrame dengan rating
    
    Returns:
    surprise.Dataset: Dataset siap untuk training
    """
    # Tentukan rentang rating
    reader = Reader(rating_scale=(0, 5))
    
    # Buat dataset Surprise
    data = Dataset.load_from_df(
        df[['user_id', 'tempat_wisata', 'rating']], 
        reader
    )
    
    return data

# Fungsi untuk melatih model
def train_recommendation_model(data, algorithm=SVD(), test_size=0.2):
    """
    Melatih model rekomendasi
    
    Parameters:
    data (surprise.Dataset): Dataset yang sudah dipersiapkan
    algorithm (surprise.prediction_algorithms): Algoritma prediksi
    test_size (float): Proporsi data untuk testing
    
    Returns:
    tuple: Model yang dilatih, skor akurasi
    """
    # Bagi data menjadi training dan testing
    trainset, testset = train_test_split(data, test_size=test_size)
    
    # Latih model
    model = algorithm
    model.fit(trainset)
    
    # Evaluasi model
    predictions = model.test(testset)
    
    # Hitung metrik akurasi
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    return model, rmse, mae

# Fungsi utama untuk melatih dan menyimpan model
def main():
    # Path file
    RECOMMENDATION_DATA_PATH = "Data Sistem Rekomendasi new.csv"
    MODEL_SAVE_PATH = "RecommenderSystem_model.sav"
    
    print("🚀 Memulai Pelatihan Model Rekomendasi Wisata")
    
    # Load data
    df = load_data(RECOMMENDATION_DATA_PATH)
    
    if df is None:
        print("❌ Gagal memuat data")
        return
    
    # Tampilkan informasi dataset
    print("\n📊 Statistik Dataset:")
    print(f"Jumlah entri: {len(df)}")
    print(f"Jumlah unique user: {df['user_id'].nunique()}")
    print(f"Jumlah unique tempat wisata: {df['tempat_wisata'].nunique()}")
    print(f"Rentang rating: {df['rating'].min()} - {df['rating'].max()}")
    
    # Persiapkan data untuk Surprise
    data = prepare_surprise_data(df)
    
    # Latih model
    try:
        print("\n🏋️ Melatih Model SVD...")
        model, rmse, mae = train_recommendation_model(data)
        
        # Tampilkan hasil evaluasi
        print("\n📈 Hasil Evaluasi Model:")
        print(f"Root Mean Square Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        
        # Simpan model
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"\n💾 Model disimpan di {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"❌ Kesalahan dalam pelatihan model: {e}")

if __name__ == "__main__":
    main()

# Contoh cara menggunakan model yang sudah dilatih
def test_model():
    # Muat model yang sudah dilatih
    loaded_model = joblib.load("RecommenderSystem_model.sav")
    
    # Contoh prediksi rating
    user_id = 25  # Ganti dengan user ID yang ingin diprediksi
    tempat_wisata = "Jember Town Square"  # Ganti dengan tempat wisata
    
    # Prediksi rating
    prediction = loaded_model.predict(user_id, tempat_wisata)
    print(f"\n🔮 Prediksi Rating:")
    print(f"User {user_id} untuk {tempat_wisata}: {prediction.est}")

# Uncomment baris di bawah untuk menguji model setelah dilatih
# test_model()