import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Dolan Jember", 
    page_icon="🌍", 
    layout="wide"
)

# Custom CSS untuk styling
# Custom CSS untuk styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .recommendation-card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.3s;
        background: linear-gradient(135deg, #f9f9f9, #e6e6e6); /* Default gradient */
    }
    .recommendation-card.kuliner {
        background: linear-gradient(135deg, #4e342e, #3e2723); /* Gelap untuk Kuliner */
    }
    .recommendation-card.religi {
        background: linear-gradient(135deg, #37474f, #263238); /* Gelap untuk Religi */
    }
    .recommendation-card.alam {
        background: linear-gradient(135deg, #33691e, #1b5e20); /* Gelap untuk Alam */
    }
    .recommendation-card.theme-park {
        background: linear-gradient(135deg, #880e4f, #4a148c); /* Gelap untuk Theme Park */
    }
    .recommendation-card.belanja {
        background: linear-gradient(135deg, #f57f17, #e65100); /* Gelap untuk Belanja */
    }
    .recommendation-card.sport {
        background: linear-gradient(135deg, #01579b, #002f6c); /* Gelap untuk Sport */
    }
    .recommendation-card:hover {
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# Path direktori gambar
IMAGES_DIR = os.path.join(os.getcwd(), "images")

class TravelRecommendationSystem:
    def __init__(self, model_path, recommendation_data_path, place_data_path, user_ratings_path):
        # Load model dan dataset
        self.model = joblib.load(model_path)
        self.recommendation_data = pd.read_csv(recommendation_data_path, delimiter=';')
        self.place_data = pd.read_csv(place_data_path)
        
        # Load atau buat user ratings
        if os.path.exists(user_ratings_path):
            self.user_ratings = pd.read_csv(user_ratings_path)
        else:
            self.user_ratings = pd.DataFrame(columns=[
                'user_id', 'tempat_wisata', 
                'rating_HTM', 'rating_Fasilitas', 
                'rating_Keamanan', 'rating_AksesJalan', 
                'rating_Transportasi',
                'total_weighted_rating'
            ])
        self.user_ratings_path = user_ratings_path

        # Bobot rating dengan rumus yang diberikan
        self.rating_weights = {
            'HTM': 0.20987654320,
            'Fasilitas': 0.20987654320,
            'Keamanan': 0.19753086420,
            'AksesJalan': 0.18518518520,
            'Transportasi': 0.19753086420
        }
    def get_all_places(self, user_id=None):
        """Mendapatkan semua tempat wisata dengan tanda jika pengguna pernah memberi rating."""
        places = []
        # Mengurutkan berdasarkan kolom 'gambar' atau 'tempat' sesuai dengan urutan
        self.place_data = self.place_data.sort_values(by="gambar", ascending=True)

        for _, row in self.place_data.iterrows():
            visited = False
            if user_id is not None:
                visited = not self.user_ratings[
                    (self.user_ratings['user_id'] == user_id) &
                    (self.user_ratings['tempat_wisata'] == row['tempat'])
                ].empty

            places.append({
                'name': row['tempat'],
                'category': row['kategori'],
                'image': row['gambar'],  # Nama file gambar dari kolom CSV
                'description': row['deskripsi'],
                'visited': visited  # Tambahkan status kunjungan
            })
        return places


    def get_recommendations(self, user_id, num_recommendations=5, selected_categories=None):
        try:
            # Cek apakah user sudah mengunjungi tempat
            user_visits = self.recommendation_data[
                (self.recommendation_data['user_id'] == user_id) & 
                (self.recommendation_data['rating'] != 0)
            ]

            # Jika belum pernah mengunjungi (hanya ada entry dengan rating 0)
            if len(user_visits) == 0:
                # Filter tempat yang pernah dikunjungi oleh user lain
                visited_places = self.recommendation_data[
                    self.recommendation_data['rating'] > 0
                ]['tempat_wisata'].unique()

                # Filter berdasarkan kategori jika ada
                if selected_categories and "Semua" not in selected_categories:
                    filtered_places = []
                    for place in visited_places:
                        place_info = self.place_data[self.place_data['tempat'] == place]
                        if not place_info.empty and place_info.iloc[0]['kategori'] in selected_categories:
                            filtered_places.append(place)
                    visited_places = filtered_places

                # Siapkan rekomendasi dari tempat yang sudah pernah dikunjungi
                recommendations = []
                for place in visited_places[:num_recommendations]:
                    place_info = self.place_data[self.place_data['tempat'] == place].iloc[0]
                    
                    recommendations.append({
                        'name': place,
                        'rating': 0,
                        'image': place_info['gambar'],
                        'description': place_info['deskripsi'],
                        'category': place_info['kategori']
                    })
                
                return recommendations

            # Jika sudah pernah mengunjungi, lakukan prediksi normal
            all_places = self.place_data['tempat'].unique()
            rated_places = self.recommendation_data[
                self.recommendation_data['user_id'] == user_id
            ]['tempat_wisata'].unique()

            # Filter tempat yang belum direview
            unrated_places = [place for place in all_places if place not in rated_places]

            # Filter berdasarkan kategori jika ada
            if selected_categories and "Semua" not in selected_categories:
                filtered_unrated = []
                for place in unrated_places:
                    place_info = self.place_data[self.place_data['tempat'] == place]
                    if not place_info.empty and place_info.iloc[0]['kategori'] in selected_categories:
                        filtered_unrated.append(place)
                unrated_places = filtered_unrated

            # Prediksi rating untuk tempat yang belum direview
            predictions = []
            for place in unrated_places:
                try:
                    predicted_rating = self.model.predict(user_id, place).est
                    predictions.append((place, predicted_rating))
                except Exception as e:
                    st.write(f"Error prediksi untuk {place}: {e}")

            # Urutkan prediksi
            predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

            # Siapkan rekomendasi
            recommendations = []
            for place, rating in predictions[:num_recommendations]:
                place_info = self.place_data[self.place_data['tempat'] == place].iloc[0]
                
                recommendations.append({
                    'name': place,
                    'rating': rating,
                    'image': place_info['gambar'],
                    'description': place_info['deskripsi'],
                    'category': place_info['kategori']
                })

            return recommendations

        except Exception as e:
            st.error(f"Kesalahan dalam menghasilkan rekomendasi: {e}")
            import traceback
            traceback.print_exc()
            return []

        except Exception as e:
            st.error(f"Kesalahan dalam menghasilkan rekomendasi: {e}")
            import traceback
            traceback.print_exc()
            return []
    def rate_place(self, user_id, place_name, ratings):
        """Metode untuk memberikan rating pada tempat wisata dan update CSV"""
        try:
            # Hitung total rating berbobot
            total_weighted_rating = sum(
                ratings[kriteria] * self.rating_weights[kriteria] 
                for kriteria in self.rating_weights
            )

            # Cari indeks tempat wisata di recommendation_data
            place_index = self.recommendation_data[
                self.recommendation_data['Tempat_Wisata'] == place_name
            ].index

            if len(place_index) > 0:
                place_index = place_index[0]
                
                # Update data di recommendation_data
                for kriteria, berat in self.rating_weights.items():
                    kolom = f'K{list(self.rating_weights.keys()).index(kriteria) + 1}'
                    current_rating = self.recommendation_data.loc[place_index, kolom]
                    
                    # Hitung rating baru (misalnya rata-rata dengan rating baru)
                    new_rating = (current_rating + ratings[kriteria]) / 2
                    
                    # Update di DataFrame
                    self.recommendation_data.loc[place_index, kolom] = new_rating

                # Simpan perubahan ke CSV
                self.recommendation_data.to_csv("Data Sistem Rekomendasi new.csv", index=False, sep=';')

            # Simpan rating pengguna
            existing_rating = self.user_ratings[
                (self.user_ratings['user_id'] == user_id) & 
                (self.user_ratings['tempat_wisata'] == place_name)
            ]

            if existing_rating.empty:
                # Tambah rating baru
                new_rating = pd.DataFrame({
                    'user_id': [user_id],
                    'tempat_wisata': [place_name],
                    'rating_HTM': [ratings['HTM']],
                    'rating_Fasilitas': [ratings['Fasilitas']],
                    'rating_Keamanan': [ratings['Keamanan']],
                    'rating_AksesJalan': [ratings['AksesJalan']],
                    'rating_Transportasi': [ratings['Transportasi']],
                    'total_weighted_rating': [total_weighted_rating]
                })
                self.user_ratings = pd.concat([self.user_ratings, new_rating], ignore_index=True)
            else:
                # Update rating yang ada
                index = existing_rating.index[0]
                self.user_ratings.loc[index, 'rating_HTM'] = ratings['HTM']
                self.user_ratings.loc[index, 'rating_Fasilitas'] = ratings['Fasilitas']
                self.user_ratings.loc[index, 'rating_Keamanan'] = ratings['Keamanan']
                self.user_ratings.loc[index, 'rating_AksesJalan'] = ratings['AksesJalan']
                self.user_ratings.loc[index, 'rating_Transportasi'] = ratings['Transportasi']
                self.user_ratings.loc[index, 'total_weighted_rating'] = total_weighted_rating

            # Simpan ke CSV
            self.user_ratings.to_csv(self.user_ratings_path, index=False)
            
            return total_weighted_rating

        except Exception as e:
            st.error(f"Kesalahan dalam memberi rating: {e}")
            return None
    
    def is_valid_user(self, user_id):
        """Memeriksa apakah user_id ada dalam dataset."""
        unique_users = pd.concat([
            self.user_ratings['user_id'],
            self.recommendation_data['user_id']
        ]).unique()
        return user_id in unique_users

def main():
    # Inisialisasi sistem rekomendasi
    recommender = TravelRecommendationSystem(
        "RecommenderSystem_model.sav",
        "Data Sistem Rekomendasi new.csv",
        "tempat_wisata.csv",
        "user_ratings.csv"
    )

    # Sidebar untuk filter
    st.sidebar.title("Filter Kategori")
    kategori_filter = st.sidebar.multiselect(
        "Pilih Kategori Wisata",
        ["Semua", "Kuliner", "Religi", "Alam", "Theme Park", "Belanja", "Sport"]
    )

    # Bagian Login
    st.title("🌍 Dolan Jember")
    st.markdown("Sistem Rekomendasi Wisata Jember")

    # Login section
    user_id = st.text_input("Masukkan User ID:", help="Masukkan ID Anda untuk rekomendasi personal")
    
    # Tampilan Awal - Semua Tempat Wisata
    st.header("Jelajahi Tempat Wisata")
    
    # Ambil semua tempat wisata
    all_places = recommender.get_all_places()
    
    # Filter kategori
    if "Semua" in kategori_filter or not kategori_filter:
        filtered_places = all_places
    else:
        filtered_places = [place for place in all_places if place['category'] in kategori_filter]

    # Tampilkan tempat wisata dalam grid
    # Tampilkan tempat wisata dalam grid
    columns = st.columns(4)

    for idx, place in enumerate(filtered_places):
        col_idx = idx % 4
        with columns[col_idx]:
            gambar_path = os.path.join(IMAGES_DIR, place['image'])  # Nama file dari kolom 'gambar'
            
            with st.expander(place['name']):
                if os.path.exists(gambar_path):  # Periksa apakah file gambar ada
                    st.image(gambar_path, use_container_width=True)
                else:
                    st.write("Gambar tidak ditemukan.")
                
                st.write(place['description'][:200] + "...")
                
                # Tombol untuk detail dan rating
                if st.button(f"Detail {place['name']}", key=f"detail_{idx}"):
                    st.write(place['description'])
                    
                    # Form rating (hanya aktif jika sudah login)
                    if user_id:
                        with st.form(key=f"rating_form_{idx}"):
                            st.write("Beri Rating:")
                            ratings = {
                                'HTM': st.slider("HTM", 1, 5, 3, key=f"htm_{idx}"),
                                'Fasilitas': st.slider("Fasilitas", 1, 5, 3, key=f"fasilitas_{idx}"),
                                'Keamanan': st.slider("Keamanan", 1, 5, 3, key=f"keamanan_{idx}"),
                                'AksesJalan': st.slider("Akses Jalan", 1, 5, 3, key=f"akses_{idx}"),
                                'Transportasi': st.slider("Transportasi", 1, 5, 3, key=f"transport_{idx}")
                            }
                            submit = st.form_submit_button("Simpan Rating")
                            
                            if submit:
                                try:
                                    user_id_int = int(user_id)
                                    total_rating = recommender.rate_place(user_id_int, place['name'], ratings)
                                    if total_rating is not None:
                                        st.success(f"Rating berhasil disimpan. Total rating: {total_rating:.2f}")
                                except ValueError:
                                    st.error("User ID harus berupa angka")






    # Bagian Rekomendasi (jika sudah login)
    if user_id:
        try:
            user_id_int = int(user_id)
            
            # Tombol untuk mendapatkan rekomendasi
            if st.button("Dapatkan Rekomendasi"):
                # Periksa validitas user
                if not recommender.is_valid_user(user_id_int):
                    st.error("Maaf, ID pengguna tidak ditemukan dalam dataset. Silakan gunakan ID yang valid.")
                else:
                    # Pass selected categories to get_recommendations
                    selected_cats = None if not kategori_filter or "Semua" in kategori_filter else kategori_filter
                    recommendations = recommender.get_recommendations(user_id_int, selected_categories=selected_cats)

                    if recommendations:
                        st.header("Rekomendasi Untukmu")
                        rec_columns = st.columns(len(recommendations))

                        for idx, (col, rec) in enumerate(zip(rec_columns, recommendations)):
                            with col:
                                gambar_path = os.path.join(IMAGES_DIR, rec['image'])

                                if os.path.exists(gambar_path):
                                    st.image(gambar_path, use_container_width=True)

                                st.markdown(f"""
                                <div class="recommendation-card {rec['category'].lower().replace(' ', '-')}">
                                    <h3>{rec['name']}</h3>
                                    <p><strong>Kategori:</strong> {rec['category']}</p>
                                    <p><strong>Rating Prediksi:</strong> {rec['rating']:.2f}/5.0</p>
                                    <p>{rec['description'][:150]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("Tidak ada rekomendasi yang tersedia untuk kategori yang dipilih.")

        except ValueError:
            st.error("User ID harus berupa angka")


if __name__ == "__main__":
    main()