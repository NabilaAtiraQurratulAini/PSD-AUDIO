import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import requests
from io import StringIO
import pickle

st.title("PROYEK SAINS DATA")

data_set_description, data_file_audio, upload_data, profile = st.tabs(["Data Set Description", "Data File Audio", "Upload Data", "Profile"])

with profile:
    st.write("### Biografi Penulis")
    st.write("Nama  : Nabila Atira Qurratul Aini ")
    st.write("Nim   : 210411100066 ")
    st.write("Kelas : Proyek Sains Data A ")
    st.write("Email Pribadi : nabilatiraqurratul@gmail.com")

with data_set_description:
    st.write("### Data Set Description")
    st.write("Data set yang digunakan adalah data set sinyal audio yang telah dilakukan perhitungan statistika. Di mana data yang digunakan sebanyak 2810. Terdapat 11 fitur dalam perhitungan data ini, diantaranya yaitu Label, ZCR Mean, ZCR Median, ZCR Standar Deviasi, ZCR Kurtosis, ZCR Skewness, RMSE, RMSE Median, RMSE Standar Deviasi, RMSE Kurtosis, dan RMSE Skewness.")
    st.write("Sinyal audio emosi adalah data audio yang mengandung informasi tentang ekspresi emosi seseorang yang terdengar melalui suara, seperti intonasi suara, tempo, kecepatan bicara, pitch, dan berbagai karakteristik lainnya. Dalam konteks mata kuliah proyek sains data, analisis sinyal audio emosi adalah salah satu aplikasi penting dari pemrosesan sinyal digital dan pemelajaran mesin untuk memahami dan menggambarkan emosi dalam data suara.")

    st.write("### Sumber Data Set Kaggle ")
    st.write("Sumber data set sinyal audio emosi ini dari kaggle.")
    st.write("https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download")

    st.write("### Source Code Aplikasi Di Google Colaboratory")
    st.write("Code dari data audio yang diiunputkan ada di google colaboratory di bawah.")
    st.write("https://colab.research.google.com/drive/1BrUIwxZfO-YFvUKULkilNdRoYi1VIBG_?usp=sharing")

    st.write("### Source Code Aplikasi Di GitHub")
    st.write("Code dari data audio yang diiunputkan ada di GitHub di bawah.")
    st.write("https://github.com/NabilaAtiraQurratulAini/UTS-PSDA.git")

with data_file_audio:
    st.write("### Data Sinyal Audio")
    st.write("Berikut adalah data set sinyal audio emosi yang telah dihitung ZCR Mean, ZCR Median, ZCR Standar Deviasi, ZCR Kurtosis, ZCR Skewness, RMSE, RMSE Median, RMSE Standar Deviasi, RMSE Kurtosis, dan RMSE Skewness.")
    df = pd.read_csv('https://raw.githubusercontent.com/NabilaAtiraQurratulAini/PsdA/main/dataaudiobaru.csv')
    st.dataframe(df)

    # memisahkan fitur (X) dan label (y)
    X = df.drop('Label', axis=1)
    y = df['Label']

    # normalisasi menggunakan Min-Max scaling
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # memisahkan data menjadi data latih dan data uji (misalnya, 80% data latih, 20% data uji)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # pelatihan model KNN sebelum PCA dan pencarian akurasi terbaik
    best_accuracy_before_pca = 0
    best_k_before_pca = 0
    accuracies_before_pca = []

    for k in range(1, 31):
        knn_before_pca = KNeighborsClassifier(n_neighbors=k)
        knn_before_pca.fit(X_train, y_train)
        y_pred_before_pca = knn_before_pca.predict(X_test)
        accuracy_before_pca = accuracy_score(y_test, y_pred_before_pca)
        
        accuracies_before_pca.append(accuracy_before_pca)
        
        if accuracy_before_pca > best_accuracy_before_pca:
            best_accuracy_before_pca = accuracy_before_pca
            best_k_before_pca = k

    # menggunakan PCA untuk reduksi data
    pca = PCA(n_components=2)  # ganti jumlah komponen sesuai kebutuhan
    X_pca = pca.fit_transform(X_normalized)

    # memisahkan data hasil PCA menjadi data latih dan data uji
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # pelatihan Model KNN setelah PCA dan Pencarian Akurasi Terbaik
    best_accuracy_after_pca = 0
    best_k_after_pca = 0
    accuracies_after_pca = []

    for k in range(1, 31):
        knn_after_pca = KNeighborsClassifier(n_neighbors=k)
        knn_after_pca.fit(X_train_pca, y_train_pca)
        y_pred_after_pca = knn_after_pca.predict(X_test_pca)
        accuracy_after_pca = accuracy_score(y_test_pca, y_pred_after_pca)
        
        accuracies_after_pca.append(accuracy_after_pca)
        
        if accuracy_after_pca > best_accuracy_after_pca:
            best_accuracy_after_pca = accuracy_after_pca
            best_k_after_pca = k

    # simpan scaler.pkl dan knn.pkl di lokasi yang diinginkan
    scaler_filename = 'C:/Visual Studio Code/scaler.pkl'
    knn_filename = 'C:/Visual Studio Code/knn.pkl'

    joblib.dump(scaler, scaler_filename)
    joblib.dump(knn_after_pca, knn_filename)  # simpan model KNN setelah PCA

    # membaca data uji
    df_test = pd.read_csv('https://raw.githubusercontent.com/NabilaAtiraQurratulAini/PsdA/main/dataaudiobaru.csv')

    # memisahkan fitur (X_test) dari data uji
    X_test = df_test.drop('Label', axis=1)

    # normalisasi data uji menggunakan scaler yang sudah ada
    X_test_normalized = scaler.transform(X_test)

    # menggunakan PCA untuk reduksi data pada data uji
    X_test_pca = pca.transform(X_test_normalized)

    # membuat prediksi menggunakan model KNN terbaik setelah PCA
    y_pred_after_pca = knn_after_pca.predict(X_test_pca)

    # menambahkan kolom 'Prediksi' ke dataframe
    df_test['Prediksi setelah PCA'] = y_pred_after_pca

    # menampilkan tabel prediksi sebelum dan setelah PCA
    st.write("### Tabel Prediksi KNN")
    st.write("Tabel prediksi KNN :")
    st.dataframe(df)

    # tampilkan akurasi KNN terbaik sebelum dan setelah PCA
    st.write(f"Akurasi KNN terbaik : {best_accuracy_before_pca:.2f}")

    st.write("### Tabel Prediksi Setelah PCA")
    st.write("Tabel prediksi setelah PCA : ")

    # gabung kolom 'Label' dari df dan 'Prediksi setelah PCA' dari df_test
    combined_df = pd.concat([df['Label'], df_test['Prediksi setelah PCA']], axis=1)

    # menampilkan tabel prediksi sebelum dan setelah PCA
    st.dataframe(combined_df)

    st.write(f"Akurasi KNN terbaik setelah PCA : {best_accuracy_after_pca:.2f}")

    # buat grafik akurasi model sebelum dan setelah PCA berdasarkan tuning parameter (nilai K)
    k_values = range(1, 31)
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, accuracies_before_pca, marker='o', linestyle='-', label='Sebelum PCA')
    plt.plot(k_values, accuracies_after_pca, marker='x', linestyle='-', label='Setelah PCA')
    st.write("### Grafik Model KNN Berdasarkan Nilai K")
    plt.title('Akurasi Model KNN Berdasarkan Nilai K')
    plt.xlabel('Nilai K')
    plt.ylabel('Akurasi')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

with upload_data:
    # fungsi perhitungan statistik yang sama seperti sebelumnya
    def calculate_statistics(audio_path):
        y, sr = librosa.load(audio_path)

        # untuk menghitung nilai statistika
        mean = np.mean(y)
        std_dev = np.std(y)
        max_value = np.max(y)
        min_value = np.min(y)
        median = np.median(y)
        skewness = scipy.stats.skew(y)  # calculate skewness
        kurt = scipy.stats.kurtosis(y)  # calculate kurtosis
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1

        # untuk menghitung nilai zcr
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
        zcr_median = np.median(librosa.feature.zero_crossing_rate(y=y))
        zcr_std_dev = np.std(librosa.feature.zero_crossing_rate(y=y))
        zcr_kurtosis = scipy.stats.kurtosis(librosa.feature.zero_crossing_rate(y=y)[0])
        zcr_skew = scipy.stats.skew(librosa.feature.zero_crossing_rate(y=y)[0])

        # round the ZCR features to 3 decimal places
        zcr_mean = round(zcr_mean, 3)
        zcr_median = round(zcr_median, 3)
        zcr_std_dev = round(zcr_std_dev, 3)
        zcr_kurtosis = round(zcr_kurtosis, 3)
        zcr_skew = round(zcr_skew, 3)

        # untuk menghitung nilai rmse
        rmse = np.sqrt(np.mean(y**2))
        rmse_median = np.sqrt(np.median(y**2))
        rmse_std_dev = np.sqrt(np.std(y**2))
        rmse_kurtosis = scipy.stats.kurtosis(np.sqrt(y**2))
        rmse_skew = scipy.stats.skew(np.sqrt(y**2))

        return [zcr_mean, zcr_median, zcr_std_dev, zcr_kurtosis, zcr_skew, rmse, rmse_median, rmse_std_dev, rmse_kurtosis, rmse_skew]
    
    def main():
        # tampilan Streamlit
        st.write('### Ekstraksi Fitur Audio')
        st.write('Unggah file audio WAV untuk menghitung fitur statistiknya.')

        # unggah file audio
        uploaded_audio = st.file_uploader("Pilih file audio", type=["wav"])

        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav', start_time=0)

            # hitung statistik
            audio_features = calculate_statistics(uploaded_audio)
            
            # with open('scalerbaru.pkl', 'rb') as standarisasi:
                #loadscal= pickle.load(standarisasi)
            
            #X_uji=loadscal.transform(X_uji) # normalisasi x testing dari hasil normalisasi x train yang disimpan dalam model

            #with open('PCA8.pkl', 'rb') as pca:
                 #loadpca = pickle.load(pca)

            #X_uji=loadpca.transform(X_uji)
                #X_uji
            
            #knn = KNeighborsClassifier(n_neighbors= 13, metric = "euclidean")
                #dump(knn, open('modelknn.pkl', 'wb'))

            #y_prediksi = knn.predict(X_uji)
            #y_prediksi

            feature_names = [
                "ZCR Mean", "ZCR Median", "ZCR Std Dev", "ZCR Kurtosis", "ZCR Skew",
                "RMSE Mean", "RMSE Median", "RMSE Std Dev", "RMSE Kurtosis", "RMSE Skew"
            ]

            # normalisasi menggunakan Min-Max Scaling
            scaler = MinMaxScaler()

            # konversi audio_features ke dalam format array NumPy
            audio_features_np = np.array(audio_features)

            # ubah scaled_features menjadi matriks 2D
            scaled_features = scaler.fit_transform(audio_features_np.reshape(1, -1))

            url = "https://raw.githubusercontent.com/NabilaAtiraQurratulAini/PsdA/main/dataaudiobaru.csv"
            response = requests.get(url)
            csv_data = response.text
            
            # create a DataFrame from the CSV data
            df = pd.read_csv(StringIO(csv_data))
            df = df.round({'ZCR Mean': 3, 'ZCR Median': 3, 'ZCR Std Dev': 3, 'ZCR Kurtosis': 3, 'ZCR Skew': 3})
            
            # filter rows with matching ZCR features
            matching_rows = df[
                (df['ZCR Mean'] == audio_features[0]) &
                (df['ZCR Median'] == audio_features[1]) &
                (df['ZCR Std Dev'] == audio_features[2]) &
                (df['ZCR Kurtosis'] == audio_features[3]) &
                (df['ZCR Skew'] == audio_features[4])
            ]

            # tampilkan hasil dengan nama variabel fitur
            st.write("### Hasil Ekstraksi Fitur")
            for i, feature in enumerate(audio_features):
                st.write(f"{feature_names[i]} : {feature}")

            # tampilkan hasil normalisasi
            st.write("### Hasil Normalisasi ")
            for i in range(len(feature_names)):
                st.write(f"{feature_names[i]}: {scaled_features[0][i]}")
            
            # simpan data yang telah dinormalisasi ke dalam file pickle
            with open('normalized_audio_features.pkl', 'wb') as f:
                pickle.dump(scaled_features, f)

            if not matching_rows.empty:
                st.write("### Matching Labels")
                st.write("Label pada audio yang di inputkan adalah :")
                st.write(matching_rows['Label'].tolist())
            else:
                st.write("No matching rows found.")

            # load data normalisasi from pickle file
            with open('normalized_audio_features.pkl', 'rb') as f:
                scaled_features = pickle.load(f)

            # bagi data menjadi data pelatihan dan data pengujian
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # mencari akurasi terbaik dengan nilai k dari 1 hingga 30
            best_accuracy = 0
            best_k = 0
            best_knn_model = None

            for k in range(1, 31):
                # buat dan latih model KNN
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                
                # prediksi pada data pengujian
                y_pred = knn.predict(X_test)

                # hitung akurasi
                accuracy = accuracy_score(y_test, y_pred)
                
                # periksa apakah ini adalah akurasi terbaik yang ditemukan
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_k = k
                    best_knn_model = knn

            # tampilkan akurasi terbaik dan nilai k yang sesuai
            st.write("### Akurasi KNN")
            st.write(f"Akurasi terbaik adalah {best_accuracy} dengan nilai K = {best_k}")

            # simpan model KNN terbaik ke dalam file pickle
            with open('best_knn_model.pkl', 'wb') as model_file:
                joblib.dump(best_knn_model, model_file)
            
            # load model KNN from pickle file
            with open('best_knn_model.pkl', 'rb') as model_file:
                knn_model = joblib.load(model_file)
            
            # bagi data menjadi data pelatihan dan data pengujian
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # gunakan model KNN yang telah dimuat untuk membuat prediksi pada data pengujian
            y_pred = knn_model.predict(X_test)

            # hitung akurasi
            accuracy = accuracy_score(y_test, y_pred)

            # tampilkan akurasi terbaik dan nilai k yang sesuai
            st.write(f"Akurasi terbaik dari model pickle dengan nilai K terbaik adalah {accuracy}")
            
            # bagi data menjadi data pelatihan dan data pengujian
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # inisialisasi dan lakukan reduksi PCA
            n_components = 5  # ganti sesuai dengan jumlah komponen yang diinginkan
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # simpan hasil reduksi PCA ke dalam file pickle
            with open('pca_features.pkl', 'wb') as pca_file:
                pickle.dump((X_train_pca, X_test_pca, y_train, y_test), pca_file)
            
            # load data PCA from pickle file
            with open('pca_features.pkl', 'rb') as pca_file:
                X_train_pca, X_test_pca, y_train, y_test = pickle.load(pca_file)

            # buat dan latih model KNN pada data PCA
            knn = KNeighborsClassifier(n_neighbors=5)  # ganti dengan nilai k yang sesuai
            knn.fit(X_train_pca, y_train)

            # prediksi pada data pengujian PCA
            y_pred = knn.predict(X_test_pca)

            # hitung akurasi pada data pengujian PCA
            accuracy = accuracy_score(y_test, y_pred)

            # tampilkan akurasi
            st.write("### Akurasi PCA")
            st.write(f"Akurasi pada data PCA adalah {accuracy}")

    if __name__ == "__main__":
        main()