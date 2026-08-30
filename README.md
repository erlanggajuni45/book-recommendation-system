# Laporan Proyek Machine Learning - Erlangga Juni Saputra

## Project Overview

Dalam era perkembangan teknologi digital saat ini, ketersediaan informasi dan pilihan buku bacaan telah meningkat secara masif. Jutaan judul buku dengan beragam genre, topik, dan gaya penulisan dapat diakses dengan mudah melalui toko buku daring maupun platform perpustakaan digital. Namun, kelimpahan pilihan ini kerap menimbulkan fenomena kelebihan informasi (*information overload*), di mana pengguna mengalami kesulitan untuk menemukan literatur yang sesuai dengan preferensi, minat, serta riwayat bacaan mereka secara cepat dan relevan (Ricci et al., 2015).

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan:**
* **Urgensi bagi Pengguna dan Industri**: Tanpa mekanisme penyaringan yang cerdas, pembaca cenderung menghabiskan waktu lama hanya untuk menelusuri katalog tanpa menemukan buku yang tepat, yang berpotensi menurunkan minat bacat dan keterlibatan pengguna pada platform digital.
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
   * Menggunakan teknik representasi tteks *Term Frequency-Inverse Document Frequency* (**TF-IDF Vectorizer**) untuk mengekstrasi vektor kata kunci, kemudian menghitung derajat kesamaan antar-vektor buku menggunakan **Cosine Similarity**.
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

---

### 2. Data Preparation untuk Collaborative Filtering

* **Pengambilan Sampel Representatif (*Sampling*)**:
  * **Proses**: Mengambil sampel $100.000$ interaksi rating secara acak dari total $\approx 6$ juta baris data dengan `random_state=42`.
  * **Alasan**: Mengoptimalkan konsumsi memori dan efisiensi waktu komputasi pelatihan Deep Learning tanpa mengorbankan variasi interaksi data.
* **Encoding ID Pengguna dan Buku ke Indeks Integer**:
  * **Proses**: Memetakan `user_id` dan `book_id` unik ke dalam indeks integer berurutan (0 hinggaa $N-1$) serta membuat *reverse mapping dictionary*.
  * **Alasan**: Lapisan *Embedding* pada model jaringan saraf tiruan membutuhkan input indeks diskrit berurutan untuk menginisialisasi tabel matriks bobot representasi vektor laten.
* **Normalisasi Nilai Rating (*Min-Max Scaling*)**:
  * **Proses**: Mentransformasikan nilai rating skala 1 - 5 ke rentang angka interval [0, 1] menggunakan formula: `rating_norm = (rating - min_rating) / (max_rating - min_rating)`
  * **Alasan**: Memudahkan konvergensi fungsi optimasi dan menyesuaikan dengan fungsi aktivasi output *Sigmoid* pada model deep learning yang menghasilkan rentang skor probabilitas [0, 1].
* **Pengacakan dan Pembagian Dataset (*Train-Validation Split*)**:
  * **Proses**: Mengacak dataset interaksi dan membaginya dengan proporsi 80% data latih dan 20% data validasi.
  * **Alasan**: Menyediakan subset validasi independen yang belum pernah dilihat oleh model selama proses pelatihan untuk memantau performa *loss* dan menghindari *overfitting*.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
