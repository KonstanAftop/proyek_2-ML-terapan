# Laporan Proyek Machine Learning - Konstan Aftop Anewata Ndruru

## Project Overview

Di era digital saat ini, industri perfilman dan platform streaming tumbuh pesat, menyediakan jutaan pilihan film kepada pengguna. Namun, kelimpahan konten ini justru menciptakan masalah baru: overload informasi. Pengguna sering kali merasa kewalahan dan kesulitan menemukan film yang sesuai dengan selera atau minat mereka. Hal ini dapat menyebabkan penurunan engagement pengguna, waktu tonton yang tidak optimal, dan bahkan churn (berhenti berlangganan) karena pengguna tidak lagi merasakan nilai dari platform yang ada. Mereka menghabiskan lebih banyak waktu mencari daripada menonton, dan akhirnya mungkin tidak menemukan apa pun yang menarik.

Pentingnya mengatasi masalah ini sangat krusial bagi platform film. Sistem yang mampu menyajikan film yang relevan secara personal kepada setiap pengguna dapat meningkatkan kepuasan pengguna, memperpanjang durasi tonton, dan pada akhirnya meningkatkan retensi pelanggan. Dengan kata lain, membantu pengguna menemukan film yang mereka cintai bukan hanya tentang kenyamanan, tetapi juga tentang keberlanjutan bisnis platform itu sendiri.

Di sinilah machine learning hadir sebagai solusi yang powerful. Dengan menganalisis data historis preferensi pengguna, seperti riwayat tontonan, rating yang diberikan, genre favorit, atau bahkan interaksi pengguna lain, machine learning dapat mengidentifikasi pola-pola tersembunyi. Pola ini kemudian digunakan untuk memprediksi film apa yang kemungkinan besar akan disukai oleh pengguna tertentu. Sistem rekomendasi yang didukung machine learning dapat bertindak sebagai "kurator pribadi" bagi setiap pengguna, secara proaktif menyarankan konten yang paling sesuai dan menarik, sehingga mengubah pengalaman menonton dari yang awalnya memusingkan menjadi menyenangkan dan personal.

Pentingnya sistem rekomendasi dalam mengatasi masalah information overload dan meningkatkan pengalaman pengguna telah banyak dibahas dalam literatur. Contohnya, Aggarwal et al. (2016) dalam bukunya menjelaskan secara komprehensif berbagai teknik dan aplikasi sistem rekomendasi dalam berbagai domain, termasuk media digital, menegaskan perannya yang krusial dalam personalisasi konten [1].

### Referensi
<br>[1] Aggarwal, C. C. (2016). Recommender Systems: The Textbook. Springer.

## Business Understanding

### Problem Statements
1. Pengguna platform film kesulitan menemukan film yang sesuai dengan preferensi mereka di tengah banyaknya pilihan, yang mengakibatkan overload informasi dan membuang waktu.
2. Tingkat engagement pengguna dan waktu tonton pada platform film berpotensi rendah karena kurangnya personalisasi, sehingga pengguna cenderung tidak menjelajahi konten lebih lanjut atau kembali untuk menonton film.
### Goals
1. Mengembangkan sistem rekomendasi film yang mampu menyarankan film relevan kepada pengguna berdasarkan riwayat tontonan dan preferensi mereka, mempermudah proses penemuan konten.
2. Meningkatkan personalisasi pengalaman pengguna untuk mendorong engagement dan waktu tonton, dengan menyediakan rekomendasi film yang secara proaktif menarik minat mereka.
### Solution Approach
Untuk membangun sistem rekomendasi film yang personal dan relevan, kami mengusulkan dua pendekatan utama:
1. Content-Based Filtering:
Merekomendasikan film berdasarkan kemiripan atribut film (genre, sutradara, aktor, sinopsis) dengan film yang disukai pengguna sebelumnya.
Cara Kerja: Menganalisis metadata film dan menghitung kemiripan (misalnya, cosine similarity) untuk menemukan film serupa.
2. Collaborative Filtering Model Based Deep Learning:
Memanfaatkan deep learning untuk mempelajari pola interaksi kompleks antara pengguna dan film dari data rating atau tontonan.
Cara Kerja: Menggunakan arsitektur jaringan saraf (seperti Neural Collaborative Filtering atau Autoencoders) untuk memodelkan preferensi pengguna dan memprediksi film yang akan disukai, menangkap hubungan non-linier yang lebih dalam.

## Data Understanding
Pada proyek ini, kami menggunakan MovieLens Latest Datasets (Small) yang diperoleh dari GroupLens melalui tautan https://grouplens.org/datasets/movielens/. Dataset ini merupakan salah satu standar benchmark yang umum digunakan dalam penelitian sistem rekomendasi. Versi "Small" dari dataset ini, yang terakhir diperbarui pada September 2018, mencakup 100.000 rating dan 3.600 aplikasi tag yang diterapkan pada 9.000 film oleh 600 pengguna.

Untuk proyek ini, saya akan fokus menggunakan tiga file utama: movies.csv, tags.csv, dan ratings.csv. File-file ini ditulis sebagai comma-separated values (CSV) dengan satu baris header dan di-encode sebagai UTF-8.

Kondisi data awal menunjukkan bahwa setiap file memiliki struktur kolom yang spesifik untuk merepresentasikan informasi film, pengguna, tag, dan rating. Kolom yang mengandung koma akan di-escape menggunakan tanda kutip ganda.

### 1. movies.csv

| Nama Variabel | Deskripsi                                             |      |
| ------------- | ----------------------------------------------------- | ---- |
| `movieId`     | ID unik untuk setiap film.                            |      |
| `title`       | Judul film, termasuk tahun rilis dalam tanda kurung.  |      |
| `genres`      | Daftar genre film yang dipisahkan oleh tanda pipa     |      |

### 2. tags.csv

| Nama Variabel | Deskripsi                                                         |
| ------------- | ----------------------------------------------------------------- |
| `userId`      | ID unik untuk setiap pengguna.                                    |
| `movieId`     | ID film yang konsisten dengan `movies.csv` dan `ratings.csv`.     |
| `tag`         | Tag yang dibuat oleh pengguna.                                    |
| `timestamp`   | Waktu saat tag diterapkan (dalam detik sejak 1 Januari 1970 UTC). |

### 3. ratings.csv
| Nama Variabel | Deskripsi                                                                |
| ------------- | ------------------------------------------------------------------------ |
| `userId`      | ID unik untuk setiap pengguna.                                           |
| `movieId`     | ID film yang konsisten dengan `movies.csv` dan `tags.csv`.               |
| `rating`      | Rating yang diberikan pengguna pada film (skala 0.5 hingga 5.0 bintang). |
| `timestamp`   | Waktu saat rating diberikan (dalam detik sejak 1 Januari 1970 UTC).      |

Rincian banyak data setiap tabel sebagai berikut :

![alt text](image.png)

### Exploratory Data Analysis
1. Tidak terdapat missing value pada ketiga dataframe yang ada.
2. Pada data tags, terdapat baris data movieId dan userId yang sama dengan tags berbeda.
3. Pada data movies, terdapat data film yang sama ditandai dengan nomor unik berbeda.
4. Distribusi data rating terkonsentras ke nilai sedang hingga tinggi.![alt text](image-1.png)
5. Genre Drama menjadi drama paling banyak di data movies, diikuti Comedy dan Thriller.![alt text](image-2.png)


## Data Preparation

Tahapan data preparation dilakukan untuk mempersiapkan data agar sesuai dengan format yang dibutuhkan oleh kedua model rekomendasi: **Content-Based Filtering** dan **Collaborative Filtering Model Based Deep Learning**.  

---
### Persiapan Data secara Umum
1. Membuang fitur 'timestamp' yang tidak relevan/tidak digunakan 
```  python
df_ratings = df_ratings.drop(columns='timestamp')
df_tags = df_tags.drop(columns='timestamp')
```
2. Menggabungkan tag dari dataframe tags dengan movieId dan userId yang sama.
``` python
df_tags_cleaned = df_tags.groupby(['movieId','userId']).agg({
    'tag' : lambda x: ' '.join(set(x))
}).reset_index()
```
3. Menghapus duplikat film dengan judul yang sama tetapi, nomor unik berbeda di dataframe movies.
``` python
df_movies_cleaned=df_movies.drop_duplicates(subset='title', keep='first')
```
4. Menggabungkan dataframe tags dan movies untuk input content based filtering, disimpan sebagai df_final.
``` python
df_rating_tags = pd.merge(df_ratings, df_tags_cleaned, on=['userId', 'movieId'], how='inner')
df_final = pd.merge(df_rating_tags, df_movies_cleaned, on=['movieId'], how='inner')
```
5. Menggabungkan dataframe ratings dan movies untuk input collaborative filtering model based.
``` python
df_final2 = pd.merge(df_ratings, df_movies_cleaned, on='movieId', how='inner')
```

### 1. Persiapan Data untuk Content-Based Filtering

Untuk model Content-Based Filtering, fokus utamanya adalah pada **atribut film** (`genres` dan `tag`) yang akan digunakan untuk mengidentifikasi kemiripan antar film.
#### a. Pembersihan dan Penggabungan Fitur Konten
- Kolom genres dibersihkan dengan menghapus karakter pemisah (|) dan menggantinya dengan spasi, memudahkan tokenisasi teks.
- Kolom genres dan tag (yang sudah digabung) kemudian digabungkan menjadi satu kolom baru bernama extracted_features. Seluruh teks diubah menjadi huruf kecil (.str.lower()) untuk memastikan konsistensi dan menghindari duplikasi karena perbedaan kapitalisasi. Kolom inilah yang nantinya akan digunakan untuk menghitung kemiripan konten antar film.
```python
df_final_content_based['genres'] = df_final_content_based['genres'].str.replace('|', ' ', regex=False)
df_final_content_based['extracted_features'] = df_final_content_based['genres'].str.lower() + ' ' + df_final_content_based['tag'].str.lower()
```
#### b. Melakukan ekstraksi fitur teks menjadi numerik dengan TF-IDF
Untuk mengubah extracted_features (yang berisi gabungan genre dan tag) menjadi representasi numerik yang dapat diolah, digunakan TfidfVectorizer. TfidfVectorizer (Term Frequency-Inverse Document Frequency) membantu menimbang pentingnya kata-kata dalam dokumen (extracted_features) relatif terhadap keseluruhan korpus. analyzer='word' memastikan tokenisasi per kata, min_df=0.0 berarti tidak ada batas minimum frekuensi dokumen untuk sebuah kata, dan stop_words='english' menghilangkan kata-kata umum dalam bahasa Inggris yang tidak informatif.
``` python
tf = TfidfVectorizer(analyzer='word',min_df=0.0, stop_words='english')
tf.fit(df_final_content_based['extracted_features'])
tf.get_feature_names_out()
content_based_tfidf_matrix = tf.fit_transform(df_final_content_based['extracted_features'])
```
#### Alasan: 
Pembersihan dan penggabungan fitur menciptakan representasi teks yang komprehensif dan konsisten. Ekstraksi fitur dengan TF-IDF mengubah teks menjadi vektor numerik yang dapat diproses model, menyoroti kata kunci relevan untuk mengidentifikasi kemiripan antar film.

### 2. Persiapan Data untuk Collaborative Filtering Model Based Deep Learning
Untuk model Collaborative Filtering berbasis Deep Learning, data perlu dipersiapkan dalam format yang memungkinkan model untuk belajar embedding pengguna dan film dari (`interaksi rating`).

#### a. Encoding ID Pengguna dan Film
- userId dan movieId yang awalnya berupa ID unik diubah menjadi indeks numerik berurutan (0, 1, 2, ...) menggunakan proses encoding. Ini penting karena model deep learning biasanya memerlukan input berupa indeks integer untuk embedding layer.
- Dibuat juga pemetaan balik dari indeks ter-encode ke ID asli (user_encoded_to_user, movie_encoded_to_movie) untuk memudahkan interpretasi hasil rekomendasi.

#### Alasan :
Proses encoding ini sangat krusial. Model deep learning tidak dapat secara langsung memproses ID string atau ID numerik yang tidak berurutan. Dengan mengubahnya menjadi indeks berurutan, kita dapat menggunakan embedding layer yang efisien untuk mempelajari representasi vektor (embedding) untuk setiap pengguna dan film, yang merupakan dasar dari model Collaborative Filtering berbasis Deep Learning.

#### b. Mengacak urutan baris (shuffling) sebelum data dibagi
``` python
df_final_collaborative = df_final_collaborative.sample(frac=1, random_state=42)
```
- frac=1 berarti mengambil 100% data (semua baris), tapi dalam urutan acak.
- random_state=42 digunakan agar hasil pengacakan konsisten (reproducible) setiap kali dijalankan.

 #### Alasan :
 Memastikan bahwa ketika data kemudian dibagi menjadi set pelatihan dan validasi (misalnya 80% untuk pelatihan dan 20% untuk validasi), distribusi data di kedua set tersebut acak dan representatif dari keseluruhan dataset. Jika data tidak diacak, mungkin ada bias dalam pembagian data, misalnya semua data dari pengguna tertentu atau film tertentu berada di satu set saja, yang dapat mempengaruhi kinerja model dan evaluasinya.

#### c. Menyiapkan matriks fitur (x) untuk input model
``` python
x = df_final_collaborative[['user_encoded','movie_encoded']].values
```
Diambil fitur user_encoded, dan movie_encoded sebagai prediktor dari rating yang diberikan oleh user terhadap suatu film.
#### Alasan : 
- user_encoded mewakili identitas pengguna dalam bentuk numerik, sehingga model bisa mempelajari pola perilaku pengguna tertentu terhadap film.
- movie_encoded mewakili identitas film dalam bentuk numerik, memungkinkan model mengenali kecenderungan film tertentu menerima rating tinggi atau rendah dari berbagai pengguna.

#### d. Melakukan normalisasi rating yang akan menjadi target (y)
Melakukan normalisasi nilai rating menjadi range [0,1] dengan formula :
$$
\text{rating}_{\text{normalized}} = \frac{\text{rating} - \text{rating}_{\min}}{\text{rating}_{\max} - \text{rating}_{\min}}
$$

#### Alasan :
Menyesuaikan skala target (y) agar sesuai dengan fungsi aktivasi sigmoid pada model deep learning, yang hanya menghasilkan output di rentang [0, 1].
#### e. Melakukan Split Data untuk Training dan Validation
- x dan y masing-masing berisi fitur dan label (rating yang telah dinormalisasi).
- train_indices menghitung 80% dari jumlah data (x.shape[0] adalah jumlah baris).
- Data kemudian dibagi menjadi dua bagian:
- Training set (x_train, y_train) → 80% pertama
- Validation set (x_val, y_val) → 20% sisanya

#### Alasan :
- Training set dipakai untuk melatih model.
- Validation set digunakan untuk mengukur performa model pada data yang tidak terlihat selama training, untuk mendeteksi overfitting atau underfitting.

## Modeling
Tahapan ini membahas pembangunan dan implementasi model sistem rekomendasi yang dirancang untuk menyelesaikan permasalahan yang telah diidentifikasi. Proyek ini mengusulkan dua solusi rekomendasi dengan pendekatan yang berbeda: Content-Based Filtering dan Collaborative Filtering Model Based Deep Learning.

### 1. Content-Based Filtering
Model Content-Based Filtering fokus pada atribut intrinsik film untuk merekomendasikan film yang serupa dengan preferensi pengguna di masa lalu.
#### Perhitungan Kemiripan 
Untuk menentukan seberapa mirip satu film dengan film lainnya berdasarkan kontennya, digunakan cosine similarity. Cosine similarity mengukur kosinus sudut antara dua vektor non-nol dan sering digunakan dalam konteks teks karena efektif dalam ruang berdimensi tinggi. Hasilnya adalah matriks kemiripan (cosine_sim) di mana setiap entri menunjukkan kemiripan antara dua film.
```python
cosine_sim = cosine_similarity(content_based_tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=df_final_content_based['title'], columns=df_final_content_based['title'])
```
#### Membangun Rekomendasi Top-N:
Fungsi rekomendasi akan mengambil judul film yang disukai pengguna sebagai input. Pertama, ia mencari indeks film tersebut dalam cosine_sim_df. Kemudian, ia mengambil semua skor kemiripan film tersebut dengan film lain, mengurutkannya dari yang paling mirip, dan mengembalikan judul N film teratas yang paling mirip.

#### Kelebihan Content-Based Filtering:
- Mampu merekomendasikan film baru yang belum pernah dilihat pengguna lain (cold start problem untuk item) selama metadata film tersedia.
- Rekomendasi mudah dijelaskan karena didasarkan pada atribut film yang jelas.
Tidak memerlukan data interaksi dari banyak pengguna untuk memulai.
#### Kekurangan Content-Based Filtering:
- Terbatas pada fitur yang tersedia. Jika fitur film kurang detail, rekomendasi mungkin kurang bervariasi.
- Kesulitan merekomendasikan film yang benar-benar baru atau di luar preferensi historis pengguna (tidak ada "eksplorasi").
#### Rekomendasi 10 Terbaik Content Based Filtering
Jika pengguna menyukai film "Sweet Charity (1969)" dengan fitur konten "comedy drama musical romance prostitution", berikut adalah 10 rekomendasi film teratas yang serupa berdasarkan kemiripan konten:
![alt text](image-4.png)


### 2. Collaborative Filtering Model Based Deep Learning
Model ini memanfaatkan arsitektur deep learning untuk mempelajari embedding pengguna dan film dari pola rating, yang kemudian digunakan untuk memprediksi rating atau memberikan rekomendasi.

#### Arsitektur Model (RecommenderNet)
Model RecommenderNet dibangun menggunakan TensorFlow Keras, mengimplementasikan pendekatan Matrix Factorization dengan embedding layers yang dioptimalkan.
#### Embedding Layers
Dua lapisan embedding terpisah digunakan untuk userId dan movieId. Setiap pengguna dan film dipetakan ke vektor representasi laten (embedding_size=50 dimensi). Inisialisasi bobot dilakukan dengan he_normal untuk stabilitas pelatihan dan ditambahkan regularizer l2 (1e-6) untuk mencegah overfitting.
#### Bias Layers
Selain embedding vektor, model juga menyertakan bias terpisah untuk setiap pengguna dan film. Bias ini menangkap tendensi rata-rata pengguna untuk memberikan rating tertentu atau tendensi rata-rata film untuk menerima rating tertentu, terlepas dari interaksi spesifik.
#### Proses call
- Input model adalah sepasang (userId, movieId).
- Vektor embedding dan bias masing-masing pengguna dan film diambil.
- Kecocokan antara vektor pengguna dan vektor film dihitung menggunakan dot product (tf.tensordot). Ini adalah inti dari matrix factorization, di mana preferensi pengguna terhadap film dimodelkan sebagai hasil perkalian vektor laten mereka.
- Nilai bias pengguna dan film ditambahkan ke hasil dot product.
- Fungsi aktivasi sigmoid diterapkan pada hasil akhir. Sigmoid mengubah nilai output ke dalam rentang [0, 1], yang sering digunakan untuk memprediksi probabilitas atau menormalkan rating agar sesuai dengan skala tertentu (misalnya, jika rating asli 1-5, ini bisa diskalakan kembali setelah prediksi).

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(SEED)
class RecommenderNet(tf.keras.Model):
 
  # Inisialisasi model
  def __init__(self, num_users, num_movies, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    
    self.num_users = num_users
    self.num_movies = num_movies
    self.embedding_size = embedding_size

    # Layer embedding untuk user
    self.user_embedding = layers.Embedding(
        input_dim=num_users,
        output_dim=embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-4)
    )

    # Layer bias untuk user
    self.user_bias = layers.Embedding(
        input_dim=num_users,
        output_dim=1
    )

    # Layer embedding untuk film
    self.movies_embedding = layers.Embedding(
        input_dim=num_movies,
        output_dim=embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-4)
    )

    # Layer bias untuk film
    self.movies_bias = layers.Embedding(
        input_dim=num_movies,
        output_dim=1
    )
 
  # Proses forward (prediksi)
  def call(self, inputs):
    # Ambil vektor embedding dan bias berdasarkan user dan film
    user_vector = self.user_embedding(inputs[:, 0])       # vektor user
    user_bias = self.user_bias(inputs[:, 0])              # bias user
    movies_vector = self.movies_embedding(inputs[:, 1])   # vektor film
    movies_bias = self.movies_bias(inputs[:, 1])          # bias film

    # Hitung kecocokan (dot product) antara user dan film
    dot_user_movies = tf.tensordot(user_vector, movies_vector, axes=2)

    # Tambahkan bias
    x = dot_user_movies + user_bias + movies_bias
    
    # Aktivasi sigmoid untuk mengubah output ke rentang [0, 1]
    return tf.nn.sigmoid(x)
  
model = RecommenderNet(num_users, num_movies, 30) # inisialisasi model
 
# model compile
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

#### Loss function dan Optimizer
Model dikompilasi dengan loss = tf.keras.losses.BinaryCrossentropy(), yang cocok jika target rating dinormalisasi ke 0-1 atau dipandang sebagai probabilitas. Optimizer yang digunakan adalah keras.optimizers.Adam(learning_rate=0.001), yang merupakan pilihan umum dan efektif untuk melatih model deep learning. Metrik evaluasi yang digunakan selama pelatihan adalah RootMeanSquaredError (RMSE).
#### Membangun Rekomendasi Top-N
Setelah model dilatih, untuk merekomendasikan film kepada pengguna tertentu, model akan memprediksi rating yang mungkin diberikan pengguna tersebut untuk semua film yang belum pernah ditontonnya. Film-film ini kemudian diurutkan berdasarkan prediksi rating tertinggi, dan N film teratas disajikan sebagai rekomendasi.
#### Kelebihan Collaborative Filtering Model Based Deep Learning:
- Mampu menangkap interaksi dan pola preferensi yang kompleks dan non-linier antar pengguna dan film.
- Tidak memerlukan informasi konten item, cukup data interaksi (rating).
- Dapat menemukan item yang menarik yang mungkin tidak serupa secara konten (serendipitous recommendations).
#### Kekurangan Collaborative Filtering Model Based Deep Learning:
- Menghadapi masalah cold start untuk pengguna dan film baru yang belum memiliki riwayat interaksi.
- Membutuhkan banyak data interaksi untuk belajar pola yang efektif.
- Interpretasi rekomendasi bisa lebih sulit karena didasarkan pada representasi laten yang abstrak.
#### Rekomendasi 10 Terbaik Collaborative Filtering Model Based Deep Learning
![alt text](image-5.png)

## Evaluation
Pada bagian ini, evaluasi dilakukan untuk mengukur kinerja kedua model rekomendasi yang telah dibangun: Content-Based Filtering dan Collaborative Filtering Model Based Deep Learning. Metrik evaluasi yang dipilih disesuaikan dengan karakteristik masing-masing model dan tujuan proyek.

### Metrik Evaluasi

1.  **_Precision@K_**:
    * **Metrik ini digunakan untuk mengevaluasi model _Content-Based Filtering_.**
    * **Formula:** $\text{Precision@K} = \frac{\text{Jumlah item relevan di top K rekomendasi}}{\text{K}}$
    * **Cara Kerja:** _Precision@K_ mengukur proporsi rekomendasi yang relevan di antara $K$ item teratas yang disarankan oleh sistem. Dalam konteks sistem rekomendasi film, item relevan didefinisikan berdasarkan kesamaan _genre_ atau _tag_ dengan film yang disukai pengguna. Semakin tinggi nilai _Precision@K_, semakin baik model dalam merekomendasikan film yang benar-benar sesuai dengan minat pengguna.

2.  **_Root Mean Squared Error_ (RMSE)**:
    * **Metrik ini digunakan untuk mengevaluasi model _Collaborative Filtering Model Based Deep Learning_.**
    * **Formula:**
        $$ \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2} $$
        Dimana:
        * $N$ adalah jumlah prediksi.
        * $y_i$ adalah _rating_ sebenarnya (nilai target).
        * $\hat{y}_i$ adalah _rating_ yang diprediksi oleh model.
    * **Cara Kerja:** RMSE mengukur besarnya selisih antara nilai rating yang diprediksi oleh model dan nilai rating sebenarnya. Ini adalah metrik yang umum digunakan untuk masalah regresi (prediksi nilai numerik). Semakin rendah nilai RMSE, semakin akurat prediksi _rating_ yang dihasilkan oleh model. 

### Hasil Proyek Berdasarkan Metrik Evaluasi
#### Hasil Content-Based Filtering (Precision@10):
Untuk evaluasi Content-Based Filtering, kami mengambil sampel film "Sweet Charity (1969)" dengan genre dan tag sebagai acuan (comedy drama musical romance prostitution). Rekomendasi dianggap relevan jika terdapat minimal satu bagian dari genre atau tag yang sesuai dengan film acuan.

Precision@10 yang diperoleh adalah 1.0. Ini berarti bahwa dari 10 rekomendasi teratas yang diberikan oleh model Content-Based Filtering untuk film acuan, seluruh 10 rekomendasi tersebut dianggap relevan berdasarkan kriteria yang ditentukan. 

#### Hasil Collaborative Filtering Model Based Deep Learning (RMSE):

Model dilatih untuk memprediksi rating pengguna terhadap film. Evaluasi dilakukan pada data validasi/uji.
Hasil: Nilai Root Mean Squared Error (RMSE) yang dicapai adalah 0.1936.
Interpretasi Hasil
Model Content-Based Filtering menunjukkan kinerja yang sangat baik dengan Precision@10 sebesar 1.0. Ini mengindikasikan bahwa model ini sangat efektif dalam merekomendasikan film-film yang secara konten sangat mirip dengan film yang sudah disukai pengguna. Hasil ini diharapkan mengingat definisi relevansi yang didasarkan pada kesamaan genre atau tag. Model ini sangat kuat dalam menyediakan rekomendasi "eksplisit" berdasarkan atribut item.
Model Collaborative Filtering Model Based Deep Learning mencapai RMSE sebesar 0.1936. Mengingat bahwa rating telah dinormalisasi ke rentang [0, 1] (karena penggunaan sigmoid di output layer), nilai RMSE sebesar 0.1936 menunjukkan bahwa rata-rata kesalahan prediksi rating oleh model berada di sekitar 0.19 dari skala 0-1. Ini adalah nilai yang cukup rendah dan mengindikasikan bahwa model memiliki kemampuan yang baik dalam memprediksi preferensi rating pengguna. Kemampuan prediksi rating yang akurat ini secara langsung berkorelasi dengan kualitas rekomendasi top-N yang dihasilkan.
#### Kesimpulan Hasil Rekomendasi
Secara keseluruhan, kedua model menunjukkan kinerja yang menjanjikan dalam domain masing-masing. Content-Based Filtering sangat baik dalam merekomendasikan item serupa secara langsung, sementara Collaborative Filtering Model Based Deep Learning menunjukkan akurasi yang solid dalam memprediksi preferensi pengguna secara lebih abstrak melalui pola interaksi. Kombinasi kedua pendekatan ini berpotensi memberikan sistem rekomendasi yang komprehensif dan efektif.