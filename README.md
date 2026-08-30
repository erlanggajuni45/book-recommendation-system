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
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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
