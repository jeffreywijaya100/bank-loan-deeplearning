# Deep Learning for Bank Loan Approval Prediction — Backpropagation Neural Network
**Binary classification to predict whether a person is granted a personal loan(1) or not(0) using BPNN.**

Pada proyek ini, saya berperan sebagai data scientist di salah satu bank terbesar di Indonesia dengan tanggung jawab mengembangkan sistem kecerdasan buatan untuk membantu perusahaan menentukan apakah pengajuan pinjaman seorang nasabah layak diterima atau ditolak. Solusi ini diharapkan mampu meningkatkan efektivitas proses underwriting secara otomatis berdasarkan karakteristik dan profil calon pemohon pinjaman. Sistem ini juga bertujuan untuk meningkatkan akurasi pengambilan keputusan kredit dan meminimalkan risiko kredit macet.

## Tahapan dan Pendekatan Proyek

### 1. Data Understanding & Data Preprocessing

Dataset yang diberikan memiliki sejumlah permasalahan data, di antaranya:

  - Missing values pada beberapa kolom → ditangani dengan imputasi berbasis mean/median untuk numerikal dan modus untuk kategori
  
  - Outliers pada variabel numerik tertentu → dilakukan penanganan dengan winsorization dan IQR
  
  - Ketidakkonsistenan skala variabel → dilakukan normalisasi / standard scaling
  
  - Fitur kategorikal → ditangani dengan one-hot encoding
  
  - Masalah class imbalance (jumlah pemohon yang ditolak lebih tinggi dibanding diterima) → diselesaikan dengan pendekatan SMOTE (Synthetic Minority Oversampling Technique)
  
Pendekatan ini dipilih untuk memastikan model deep learning dapat mengenali pola data secara stabil dan mencegah bias terhadap kelas mayoritas.

### 2. Train / Val / Test Split

Dataset dibagi menggunakan proporsi:

  - 80% → training

  - 10% → validation

  - 10% → test
  
Pembagian ini bertujuan memastikan model dapat dilatih secara optimal dan dievaluasi secara adil pada unseen data.

### 3. Tantangan Data Tabular

Karakter data tabular menghadirkan sejumlah kesulitan, antara lain:

  - Kualitas data (missing, outlier, noisy)
  
  - Keberagaman jenis fitur (numerikal & kategorikal)
  
  - Skala fitur yang tidak seragam
  
  - Distribusi label target yang tidak seimbang

Semua tantangan tersebut diatasi melalui preprocessing dan penyesuaian arsitektur model.

### 4. Baseline BPNN Architecture

Baseline Backpropagation Neural Network dibangun dengan spesifikasi:

  - Input layer: n nodes (jumlah fitur)
  
  - Hidden layer 1: 2 × n nodes — ReLU
  
  - Hidden layer 2: 2 × n nodes — ReLU
  
  - Output layer: num_class (binary classification)

Arsitektur ini berfungsi sebagai titik awal untuk mengukur performa awal.

### 5. Model Optimization

Setelah memperhatikan performa baseline, arsitektur dimodifikasi untuk memperoleh kinerja optimal melalui:

  - Penyesuaian jumlah layer & neuron
  
  - Penambahan dropout layer untuk mencegah overfitting
  
  - Penyesuaian learning rate
  
  - Tuning batch size dan epoch
  
  - Pemilihan optimizer terbaik (Adam)

Pendekatan ini dilakukan sebagai respons terhadap kompleksitas data tabular serta tantangan class imbalance dan variasi skala fitur.

### 6. Model Evaluation

Kinerja kedua arsitektur (baseline & optimized) diukur pada test set menggunakan:

  - Accuracy
  
  - Precision
  
  - Recall
  
  - F1-Score


  a. Evaluasi Model 1 (Arsitektur Baseline)
  - Accuracy: 98%, yang menunjukkan bahwa model baseline dapat mengklasifikasikan data dengan baik secara keseluruhan.
  - Precision dan Recall Kelas 0: Precision: 0.99 dan Recall: 0.98, menunjukkan bahwa model sangat baik dalam mengklasifikasikan kelas mayoritas (kelas 0) dengan sedikit kesalahan.    
  - Precision dan Recall Kelas 1: Precision: 0.85 dan Recall: 0.94, yang menunjukkan bahwa model kadang salah mengklasifikasikan kelas 1 sebagai kelas 0, tetapi masih cukup baik dalam mendeteksi kelas minoritas ini.
  - F1-Score: Kelas 0 memiliki F1-score yang sangat tinggi (0.99), sedangkan kelas 1 memiliki F1-score 0.89, yang lebih rendah tetapi cukup baik.
  - Macro Avg dan Weighted Avg: Macro average dan weighted average F1-scores masing-masing adalah 0.94 dan 0.98. Ini menunjukkan bahwa baseline model sedikit kurang optimal dalam menangani kelas minoritas.
  
  b. Evaluasi Model 2 (Modifikasi Arsitektur)
  - Accuracy: 99%, menunjukkan peningkatan performa yang signifikan dibanding model baseline.
  - Precision dan Recall Kelas 0: Precision dan Recall untuk kelas 0 masing-masing adalah 1.00 dan 0.99, yang menunjukkan bahwa model dapat mengklasifikasikan kelas mayoritas dengan hampir sempurna.
  - Precision dan Recall Kelas 1: Precision meningkat menjadi 0.91, dan recall menjadi 1.00, menunjukkan bahwa model sangat baik dalam mendeteksi kelas minoritas tanpa melewatkan satu pun instance dari kelas tersebut.
  - F1-Score: Kelas 1 memiliki F1-score yang meningkat menjadi 0.95 dibandingkan baseline. Ini menunjukkan kemampuan model yang lebih seimbang dalam klasifikasi, terutama pada kelas minoritas.
  - Macro Avg dan Weighted Avg: Macro average F1-score meningkat menjadi 0.97, dan weighted average tetap tinggi pada 0.99, menunjukkan perbaikan kinerja keseluruhan model.

Hasil evaluasi menunjukkan peningkatan signifikan pada F1-score dan recall setelah optimasi, yang berarti sistem lebih mampu mendeteksi calon debitur yang benar-benar layak diberi pinjaman tanpa mengabaikan risiko.

Model modifikasi arsitektur berhasil meningkatkan performa pada kelas minoritas (kelas 1) dengan kenaikan pada precision dan recall tanpa mengorbankan akurasi di kelas mayoritas (kelas 0). Perbaikan ini menghasilkan F1-score yang lebih tinggi pada kelas minoritas dan makro average yang lebih seimbang, membuat model lebih efektif dalam menangani ketidakseimbangan kelas. Modifikasi yang dilakukan, seperti tuning parameter dan penggunaan optimizer yang berbeda, terbukti membantu model dalam menangani tantangan data tabular yang kompleks dan meningkatkan generalisasi.


## 💼 Real Business Impact

Implementasi BPNN untuk loan approval prediction memberikan dampak nyata pada operasional bank:

| Impact Bisnis                      | Penjelasan                                                                |
| --------------------------------- | ------------------------------------------------------------------------- |
| 🔹 Mengurangi risiko kredit macet | Model mengenali calon debitur berkualitas dengan lebih presisi            |
| 🔹 Mempercepat proses persetujuan | Evaluasi otomatis tanpa perlu pengecekan manual                           |
| 🔹 Efisiensi biaya operasional    | Mengurangi beban analis kredit & menghemat waktu                          |
| 🔹 Customer Experience meningkat  | Peminjam menerima keputusan lebih cepat & transparan                      |
| 🔹 Mendukung compliance           | Keputusan berbasis data mengurangi bias manusia                           |
| 🔹 Scalable                       | Sistem bisa diintegrasikan ke API / core banking untuk screening otomatis |

