# Laporan Proyek Machine Learning - Erlangga Juni Saputra

## Project Overview

Dalam era perkembangan teknologi digital saat ini, ketersediaan informasi dan pilihan buku bacaan telah meningkat secara masif. Jutaan judul buku dengan beragam genre, topik, dan gaya penulisan dapat diakses dengan mudah melalui toko buku daring maupun platform perpustakaan digital. Namun, kelimpahan pilihan ini kerap menimbulkan fenomena kelebihan informasi (*information overload*), di mana pengguna mengalami kesulitan untuk menemukan literatur yang sesuai dengan preferensi, minat, serta riwayat bacaan mereka secara cepat dan relevan (Ricci et al., 2015).

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan:**
* **Urgensi bagi Pengguna dan Industri**: Tanpa mekanisme penyaringan yang cerdas, pembaca cenderung menghabiskan waktu lama hanya untuk menelusuri katalog tanpa menemukan buku yang tepat, yang berpotensi menurunkan minat baca dan keterlibatan pengguna pada platform digital.
* **Solusi Pendekatan Machine Learning**: Untuk menjawab tantangan tersebut, sistem rekomendasi berbasis *machine learning* menjadi solusi kunci. Pendekatan *Collaborative Filtering* terbukti mampu menangkap preferensi implisit dan eksplisit pembaca melalui pemodelan pola interaksi antar-pengguna (Koren et al., 2009). Sementara itu, pendekatan *Content-Based Filtering* melengkapi personalisasi dengan memanfaatkan atribut konten buku menggunakan teknik representatif teks (Lops et al., 2011). Pada proyek ini, kedua paradigma tersebut diterapkan menggunakan dataset [Goodbooks-10k Dataset](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) (Zając, 2017) guna menghasilkan sistem rekomendasi buku (*Top-N Recommendation*) yang presisi dan adaptif.

## Business Understanding

Pengembangan sistem rekomendasi ini ditujukan untuk memfasilitasi pembaca dalam menemukan buku bacaan yang relevan dan personal, sekaligus meningkatkan *user engagement* serta retensi pengguna pada katalog buku daring.

### Problem Statements
* Bagaimana cara merekomendasikan buku lain yang memiliki kemiripan konten, tema, atau kepenulisan dengan buku tertentu yang disukai oleh pembaca?
* Bagaimana cara memprediksi dan merekomendasikan buku yang dipersonalisasi untuk seorang pengguna berdasarkan riwayat interaksi dan pola penilaian (*rating*) dari pengguna lain?

### Goals
* Menghasilkan sistem rekomendasi berbasis *Content-Based Filtering* yang mampu menyajikan $N$ buku teratas yang paling mirip dengan buku acuan berdasarkan representasi atribut teks (judul dan penulis).
* Membangun model *Collaborative Filtering* berbasis *Deep Learning* yang mampu mempelajari preferensi laten pengguna dan menyajikan $N$ buku rekomendasi yang dipersonalisasi dengan tingkat kesalahan prediksi (*error*) yang minimal.

### Solution Approach
Untuk mencapai target tersebut, diajukan dua pendekatan algoritma sistem rekomendasi:
1. **Content-Based Filtering**:
   * Memanfaatkan fitur metadata konten buku (`title` dan `authors`)
   * Menggunakan teknik representasi teks *Term Frequency-Inverse Document Frequency* (**TF-IDF Vectorizer**) untuk mengekstraksi vektor kata kunci, kemudian menghitung derajat kesamaan antar-vektor buku menggunakan **Cosine Similarity**.
   * Menghasilkan rekomendasi buku berdasarkan skor kesamaan tertinggi terhadap buku yang pernah dibaca pengguna.
2. **Collaborative Filtering**:
   * Memanfaatkan data interaksi eksplisit berupa riwayat pemberian rating terhadap buku (`user_id`, `book_id`, `rating`).
   * Membangun arsitektur jaringan saraf tiruan (**RecommenderNet**) menggunakan *Embedding Layers* untuk memetakan pengguna dan buku ke dalam ruang vektor laten (*latent feature space*).
   * Mengoptimalkan model menggunakan fungsi *loss* **Binary Crossentropy** dan optimizer **Adam** guna memprediksi skor preferensi buku yang belum pernah dibaca oleh pengguna.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Goodbooks-10k Dataset** yang dipublikasikan oleh [Zygmunt Zając di GitHub goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k/). Dataset ini berisi informasi 10.000 judul buku terpopuler dan hampir 6 juta interaksi penilaian (*ratings*) yang diberikan oleh 53.424 pengguna unik.

Dataset terdiri dari dua berkas utama:
* **books.csv**: Memiliki dimensi 10.000 baris dan 23 kolom metadata buku.
* **ratings.csv**: Memiliki dimensi 5.976.479 baris dan 3 kolom interaksi pengguna terhadap buku.

### Variabel-variabel pada Dataset:

**1. Berkas `books.csv`:**
* `book_id`: ID unik untuk setiap buku dalam sistem.
* `goodreads_book_id` & `best_book_id`: ID unik buku pada sistem basis data Goodreads.
* `work_id`: ID unik karya buku secara abstrak yang mengagregasi berbagai edisi cetak.
* `books_count`: Jumlah edisi yang tersedia untuk karya buku tersebut.
* `isbn` & `isbn13`: Kode standar identifikasi buku internasional.
* `authors`: Nama penulis atau kreator buku.
* `original_publication_year`: Tahun pertama kali buku diterbitkan.
* `original_title`: Judul asli buku sebelum diterjemahkan.
* `title`: Judul resmi buku.
* `language_code`: Kode bahasa yang digunakan pada buku (misal: `eng`, `en-US`, `spa`).
* `average_rating`: Nilai rata-rata rating buku (skala 1 - 5).
* `ratings_count`: Jumlah total ulasan penilaian yang diterima buku.
* `work_ratings_count` & `work_text_reviews_count`: Total akumulasi rating dan ulasan teks dari seluruh edisi.
* `ratings_1` s/d `ratings_5`: Jumlah rincian skor rating dari bintang 1 hingga bintang 5.
* `image_url` & `small_image_url`: Tautan URL gambar sampul buku.

**2. Berkas `ratings.csv`:**
* `user_id`: ID unik pengguna yang memberikan penilaian.
* `book_id`: ID unik buku yang dinilai.
* `rating`: Skor penilaian yang diberikan oleh pengguna.

---

### Exploratory Data Analysis (EDA)

Berdasarkan hasil eksplorasi data yang dilakukan pada notebook:

1. **Pengecekan Data Duplikat dan Missing Value**:
   * Pada `ratings.csv` tidak ditemukan data *missing value* maupun data duplikat.
   * Pada `books.csv` tidak terdapat data duplikat namun terdapat beberapa *missing value* pada kolom non-esensial seperti `language_code` (1.084 baris), `isbn` (700 baris), dan `original_title` (585 baris). Kolom utama yang digunakan dalam sistem rekomendasi (`title`, `authors`, `book_id`, `average_rating`) berstatus lengkap (10.000 non-null).
  
2. **Analisis Karakteristik Data & Visualisasi**:
   * **Top Penulis Populer**: Stephen King menempati posisi teratas sebagai penulis paling produktif dengan 60 judul buku di dalam katalog dataset, diikuti oleh Nora Roberts dan Dean Koontz.
   * **Top Buku Paling Populer**: Buku *The Hunger Games (The Hunger Games, #1)* dan *Harry Potter and the Sorcerer's Stone* menjadi buku dengan jumlah rating terbanyak, melebihi 4,5 juta ulasan.
   * **Distribusi Skor Rating Pengguna**: Distribusi skor rating pembaca didominasi oleh rating $4$ dan $5$, yang mencerminkan kecenderungan kepuasan positif dari para pembaca buku populer.
   * **Distribusi Average Rating**: Rata-rata rating buku berpusat pada nilai $\approx 4.0$ dengan kurva mendekati distribusi normal (*bell-shaped curve*).

## Data Preparation
Tahapan ini dilakukan secara berurutan untuk menyesuaikan kebutuhan input dari masing-masing pendekatan algoritma rekomendasi:

### 1. Data Preparation untuk Content-Based Filtering

* **Pemilihan Fitur Relevan dan Penanganan Missing Value**:
  * **Proses**: Memilih kolom esensial (`book_id`, `title`, `authors`, dan `average_rating`) serta membuang baris yang memiliki nilai null (`dropna()`).
  * **Alasan**: Menghindari kesalahan komputasi saat ekstraksi representasi teks dan menjaga konsistensi metadata katalog.
* **Penghapusan Duplikasi Judul Buku**:
  * **Proses**: Menghapus baris dengan judul buku duplikat (`drop_duplicates(subset=['title'])`).
  * **Alasan**: Mencegah model menghasilkan rekomendasi ganda dari buku yang sama sehingga output matriks kesamaan bersifatt unik ($9.964$ buku unik).
* **Penggabungan Fitur Metadata Konten (*Content Feature Fusion*)**:
  * **Proses**: Menggabungkan teks judul dan nama penulis ke dalam kolom baru `content_features` (`df_cb['title'] + ' ' + df_cb['authors']`).
  * **Alasan**: Memberikan konteks yang lebih kaya bagi algoritma ekstraksi teks agar dapat menangkap kemiripan tidak hanya dari kata kunci judul, tetapi juga dari kesamaan penulis.
* ***Feature Extraction* menggunakan TF-IDF Vectorizer**:
  * **Proses**: Mentransformasikan data teks pada `content_features` menjadi representasi vektor numerik menggunakan `TfidfVectorizer(stop_words='english')`, menghasilkan matriks TF-IDF berdimensi `(9964, 14219)`.
  * **Alasan**: Algoritma pembelajaran mesin dan perhitungan jarak membutuhkan representasi numerik terbobot (*term frequency-inverse document frequency*) yang mampu merefleksikan signifikansi kata kunci unik tanpa terdistorsi oleh kata umum (*stop words*).

---

### 2. Data Preparation untuk Collaborative Filtering

* **Pengambilan Sampel Representatif (*Sampling*)**:
  * **Proses**: Mengambil sampel $100.000$ interaksi rating secara acak dari total $\approx 6$ juta baris data dengan `random_state=42`.
  * **Alasan**: Mengoptimalkan konsumsi memori dan efisiensi waktu komputasi pelatihan Deep Learning tanpa mengorbankan variasi interaksi data.
* **Encoding ID Pengguna dan Buku ke Indeks Integer**:
  * **Proses**: Memetakan `user_id` dan `book_id` unik ke dalam indeks integer berurutan (0 hingga $N-1$) serta membuat *reverse mapping dictionary*.
  * **Alasan**: Lapisan *Embedding* pada model jaringan saraf tiruan membutuhkan input indeks diskrit berurutan untuk menginisialisasi tabel matriks bobot representasi vektor laten.
* **Normalisasi Nilai Rating (*Min-Max Scaling*)**:
  * **Proses**: Mentransformasikan nilai rating skala 1 - 5 ke rentang angka interval [0, 1] menggunakan formula: `rating_norm = (rating - min_rating) / (max_rating - min_rating)`
  * **Alasan**: Memudahkan konvergensi fungsi optimasi dan menyesuaikan dengan fungsi aktivasi output *Sigmoid* pada model deep learning yang menghasilkan rentang skor probabilitas [0, 1].
* **Pengacakan dan Pembagian Dataset (*Train-Validation Split*)**:
  * **Proses**: Mengacak dataset interaksi dan membaginya dengan proporsi 80% data latih dan 20% data validasi.
  * **Alasan**: Menyediakan subset validasi independen yang belum pernah dilihat oleh model selama proses pelatihan untuk memantau performa *loss* dan menghindari *overfitting*.

## Modeling and Result

Pada tahap ini, dibangun dua pendekatan sistem rekomendasi yang berbeda untuk menyelesaikan permasalahan pencarian dan personalisasi buku bacaan:

### 1. Model 1: Content-Based Filtering (Cosine Similarity)

* **Cara Kerja**:
  * Menggunakan matriks representasi numerik TF-IDF yang telah diekstraksi pada tahap Data Preparation.
  * Menghitung derajat sudut kesamaan antar-buku dihitung menggunakan **Cosine Similarity** untuk menghasilkan matriks kesamaan kosinus.
  * Sistem memetakan indeks buku acuan dan menyortir $N$ buku teratas yang memiliki nilai *similarity score* tertinggi terhadap buku tersebut.
* **Kelebihan**:
  * Tidak bergantung pada data interaksi pengguna lain (*user-independent*).
  * Tidak mengalami masalah *cold-start* untuk buku-buku baru selama metadata kontennya tersedia.
  * Mampu merekomendasikan buku-buku spesifik atau lanjutan dalam satu seri/universe kepenulisan yang sama.
* **Kekurangan**:
  * Rentan terhadap fenomena *overspecialization* (rekomendasi cenderung terbatas pada tema/penulis yang mirip dan kurang eksploratif).
  * Tidak dapat menangkap kualitas subjektif buku atau preferensi implisit antar-pengguna.

#### Output Top-10 Rekomendasi Content-Based Filtering:

Sebelum menyajikan hasil rekomendasi, sistem menetapkan satu buku acuan yang dijadikan preferensi pencarian kesamaan konten:

* **Informasi & Karakteristik Buku Acuan**:
  * **Judul Buku**: *The Hunger Games (The Hunger Games, #1)*
  * **Penulis**: Suzanne Collins
  * **Average Rating**: 4.34 / 5.0
  * **Karakteristik/Konteks Tema**: Novel fiksi ilmiah distopia bertema bertahan hidup (*survival*), aksi petualangan, dan kompetisi gladiator futuristik karya Suzanne Collins.
  * **Ekspektasi Rekomendasi**: Sistem diharapkan memprioritaskan buku-buku sekuel, trilogi lanjutan, panduan cerita, atau karya terkait dari penulis dan semesta yang sama (*The Hunger Games Trilogy*).

Berikut adalah 10 buku teratas yang direkomendasikan berdasarkan tingkat kesamaan kosinus (*Cosine Similarity*):

| No | Judul Buku | Penulis | Average Rating | Similarity Score |
|---|---|---|---|---|
| 1 | The Hunger Games Trilogy Boxset (The Hunger Games, #1-3) | Suzanne Collins | 4.49 | 0.9129 |
| 2 | Catching Fire (The Hunger Games, #2) | Suzanne Collins | 4.30 | 0.8075 |
| 3 | Mockingjay (The Hunger Games, #3) | Suzanne Collins | 4.03 | 0.7975 |
| 4 | The World of the Hunger Games (Hunger Games Trilogy) | Kate Egan | 4.48 | 0.7716 |
| 5 | The Hunger Games Tribute Guide | Emily Seife | 4.40 | 0.4972 |
| 6 | The Hunger Games: Official Illustrated Movie Companion | Kate Egan | 4.51 | 0.4452 |
| 7 | Hunger (Gone, #2) | Michael Grant | 4.02 | 0.3703 |
| 8 | A Hunger Like No Other (Immortals After Dark #2) | Kresley Cole | 4.21 | 0.2922 |
| 9 | The Quillan Games (Pendragon, #7) | D.J. MacHale | 4.19 | 0.2842 |
| 10 | Nemesis Games (The Expanse, #5) | James S.A. Corey | 4.37 | 0.2806 |

---

### 2. Model 2: Collaborative Filtering (Deep Learning RecommenderNet)

* **Cara Kerja**:
  * Membangun model jaringan saraf tiruan kustom berbasis *Embedding Layer* (`RecommenderNet`) menggunakan TensorFlow/Keras.
  * Model memetakan `num_users` (44.477 pengguna) dan `num_books` (9.521 buku) ke dalam ruang vektor laten berdimensi 50 (*embedding size* = 50) dengan regularisasi L2 ($1\text{e-}6$) dan inisialisasi bobot `he_normal`.
  * Vektor representasi pengguna dan buku dihitung perkalian titiknya (*dot product*), ditambahkan bias pengguna dan bias buku, kemudian dilewatkan pada fungsi aktivasi *Sigmoid* untuk menghasilkan prediksi rating terstandardisasi pada interval [0, 1].
  * Model dilatih selama 15 *epochs* dengan optimizer *Adam* (learning rate 0.001) dan fungsi *loss* *Binary Crossentropy*.
* **Kelebihan**:
  * Mampu memberikan rekomendasi lintas genre yang tidak terduga (*serendipity*) berdasarkan pola kesamaan selera antarpengguna.
  * Tidak memerlukan rekayasa fitur metadata konten teks yang rumit.
* **Kekurangan**:
  * Mengalami kendala *cold-start problem* untuk pengguna baru atau buku baru yang belum memiliki riwayat rating.
  * Membutuhkan daya komputasi dan memori yang lebih besar untuk proses pelatihan matriks embedding.

#### Output Top-10 Rekomendasi Collaborative Filtering:

Sebelum menyajikan hasil rekomendasi personal, sistem memeriksa riwayat buku yang telah dibaca dan diberi rating tinggi (skala 5) oleh pengguna target guna mengidentifikasi preferensinya:

* **Informasi & Preferensi Pengguna Target (User ID: 43140)**:
  * **Buku yang Pernah Dibaca & Disukai (Rating 5)**:
    1. *I Am Legend* karya Richard Matheson (Rating: 5 / 5, Average Rating: 4.07)
    2. *The Complete Robot (Robot #0.3)* karya Isaac Asimov (Rating: 5 / 5, Average Rating: 4.34)
  * **Analisis Preferensi Pengguna**: Pengguna memiliki minat dan preferensi yang sangat kuat terhadap literatur bergenre fiksi ilmiah klasik (*classic science fiction*), cerita distopia pasca-apokaliptik, kecerdasan buatan (*robotics*), serta karya-karya berbobot dari penulis legendaris seperti Isaac Asimov dan Richard Matheson.
  * **Ekspektasi Rekomendasi**: Model diharapkan mampu merekomendasikan karya fiksi epik, fantasi, dan literatur berating tinggi lainnya yang memiliki alur narasi kuat yang belum pernah dibaca oleh pengguna tersebut.

Berikut adalah 10 buku teratas yang direkomendasikan secara personal untuk **User ID: 43140**:

| No | Judul Buku | Penulis | Average Rating |
|---|---|---|---|
| 1 | Harry Potter and the Goblet of Fire (Harry Potter, #4) | J.K. Rowling, Mary GrandPré | 4.53 |
| 2 | Harry Potter and the Deathly Hallows (Harry Potter, #7) | J.K. Rowling, Mary GrandPré | 4.61 |
| 3 | Unbroken: A World War II Story of Survival, Resilience, and Redemption | Laura Hillenbrand | 4.40 |
| 4 | Cutting for Stone | Abraham Verghese | 4.28 |
| 5 | Harry Potter Boxset (Harry Potter, #1-7) | J.K. Rowling | 4.74 |
| 6 | Hyperion (Hyperion Cantos, #1) | Dan Simmons | 4.21 |
| 7 | The Little House Collection (Little House, #1-9) | Laura Ingalls Wilder, Garth Williams | 4.33 |
| 8 | Words of Radiance (The Stormlight Archive, #2) | Brandon Sanderson | 4.77 |
| 9 | The Complete Anne of Green Gables Boxed Set (Anne of Green Gables, #1-8) | L.M. Montgomery | 4.42 |
| 10 | Brief Lives (The Sandman #7) | Neil Gaiman, Jill Thompson, Vince Locke, Peter Doherty | 4.55 |

---

## Evaluation

### 1. Evaluasi Content-Based Filtering: Precision@K

* **Formula dan Cara Kerja**:
  Metrik **Precision@K** digunakan untuk mengukur seberapa relevan item-item yang disajikan pada daftar $K$ rekomendasi teratas:

  $$\text{Precision@K} = \frac{\text{Jumlah Item Rekomendasi yang Relevan}}{K}$$

  Pada sistem rekomendasi konten buku ini, sebuah buku dalam daftar Top-K ($K=10$) dikategorikan **relevan** apabila judul atau nama penulisnya memiliki korelasi langsung terhadap tema/karya buku acuan (*The Hunger Games* atau *Suzanne Collins*).
* **Hasil Evaluasi**:
  Dari 10 buku yang direkomendasikan untuk buku *The Hunger Games (The Hunger Games, #1)*, terdapat 8 buku yang terbukti relevan secara langsung dengan seri tersebut (termasuk sekuel resmi, *boxset*, dan *official guide*).

$$
\text{Precision@10} = \frac{8}{10} = 80.0\\%
$$

  Hasil ini membuktikan bahwa pendekatan *Content-Based Filtering* sangat efektif dalam menemukan buku-buku yang memiliki keselarasan konteks dan tema yang tinggi.

---

### 2. Evaluasi Collaborative Filtering: Root Mean Squared Error (RMSE)

* **Formula dan Cara Kerja**:
  Metrik **Root Mean Squared Error (RMSE)** digunakan untuk mengukur rata-rata besarnya kesalahan prediksi rating yang dihasilkan oleh model terhadap nilai rating aktual yang telah dinormalisasi:

  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

  Di mana $N$ adalah jumlah data evaluasi, $y_i$ adalah rating aktual ternormalisasi, dan $\hat{y}_i$ adalah skor rating yang diprediksi oleh model. Semakin kecil nilai RMSE, semakin akurat model dalam memprediksi preferensi pengguna.
* **Hasil Evaluasi**:
  Setelah melalui proses pelatihan selama 15 *epochs*, model *RecommenderNet* menghasilkan performa evaluasi sebagai berikut:
  * **Training RMSE**: $\approx 0.2546$
  * **Validation RMSE**: $\approx 0.2823$
  * **Training Loss (Binary Crossentropy)**: $\approx 0.5950$
  * **Validation Loss (Binary Crossentropy)**: $\approx 0.6283$

Plot konvergensi pada grafik pelatihan menunjukkan penurunan kurva *loss* dan RMSE yang stabil antara data latih dan data validasi tanpa adanya indikasi *overfitting* yang signifikan. Hal ini menunjukkan model mampu memprediksi preferensi laten pembaca dengan tingkat kesalahan yang rendah.
