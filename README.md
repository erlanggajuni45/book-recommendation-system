# Laporan Proyek Machine Learning - Erlangga Juni Saputra

## Project Overview

Dalam era perkembangan teknologi digital saat ini, ketersediaan informasi dan pilihan buku bacaan telah meningkat secara masif. Jutaan judul buku dengan beragam genre, topik, dan gaya penulisan dapat diakses dengan mudah melalui toko buku daring maupun platform perpustakaan digital. Namun, kelimpahan pilihan ini kerap menimbulkan fenomena kelebihan informasi (*information overload*), di mana pengguna mengalami kesulitan untuk menemukan literatur yang sesuai dengan preferensi, minat, serta riwayat bacaan mereka secara cepat dan relevan (Ricci et al., 2015).

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan:**
* **Urgensi bagi Pengguna dan Industri**: Tanpa mekanisme penyaringan yang cerdas, pembaca cenderung menghabiskan waktu lama hanya untuk menelusuri katalog tanpa menemukan buku yang tepat, yang berpotensi menurunkan minat bacat dan keterlibatan pengguna pada platform digital.
* **Solusi Pendekatan Machine Learning**: Untuk menjawab tantangan tersebut, sistem rekomendasi berbasis *machine learning* menjadi solusi kunci. Pendekatan *Collaborative Filtering* terbukti mampu menangkap preferensi implisit dan eksplisit pembaca melalui pemodelan pola interaksi antar-pengguna (Koren et al., 2009). Sementara itu, pendekatan *Content-Based Filtering* melengkapi personalisasi dengan memanfaatkan atribut konten buku menggunakan teknik representatif teks (Lops et al., 2011). Pada proyek ini, kedua paradigma tersebut diterapkan menggunakan dataset [Goodbooks-10k Dataset](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) (Zając, 2017) guna menghasilkan sistem rekomendasi buku (*Top-N Recommendation*) yang presisi dan adaptif.

## Business Understanding

Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution approach (algoritma atau pendekatan sistem rekomendasi).

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
