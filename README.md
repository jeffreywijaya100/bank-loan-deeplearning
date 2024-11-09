# Binary classification to predict whether a person is granted a personal loan(1) or not(0) using BPNN.
Charlie merupakan salah satu data scientist di sebuah bank terbesar di Indonesia. Dia diminta untuk membuat sebuah aplikasi yang membantu perusahaan untuk menentukan apakah sebuah pengajuan pinjaman (loan) yang dilakukan seseorang adalah dapat diterima atau ditolak (binary classification).

Buatlah arsitektur Back Propagation Neural Network (BPNN). Berikut adalah ketentuan yang perlu diperhatikan dalam pembuatan arsitektur BPNN.

a. Dataset yang diberikan memiliki beberapa problem, lakukan praproses data
untuk menyelesaikan problem dari data tersebut. Sebutkan problem apa saja yang kalian
temukan dari data yang diberikan, berikan penjelasan mengenai pendekatan apa yang kalian
gunakan dan kenapa memilih pendekatan yang dipilih?

b. Lakukan eksplorasi data terlebih dahulu untuk memahami permasalahan
yang dihadapi terlebih dahulu. Selanjutnya pisahkan dataset menjadi train, test dan validation
set dengan ketentuan (80 train, 10 val, 10 test)

c. Identifikasikan tantangan dan kesulitan dari data tabular yang diberikan.
Hal ini terkait dengan Kualitas Data (Data Kotor dan Hilang, Outliers), Jenis Data (Heterogenitas
Fitur, Encoding Fitur Kategorikal, Skala dan Normalisasi) dan Ketidakseimbangan Kelas (Class
Imbalance).

d. Buatlah arsitektur baseline dengan n nodes input layer, 2 buah hidden layer
dengan banyak 2 × n nodes awal dan layer akhir banyak kelas nya (n, 2 × n, 2 × n, num_class).
Keterangan: n adalah banyak input dan num_class adalah banyak kelas. Activation function
untuk tiap hidden layer menggunakan ReLU

e. Setelah mengetahui hasil dari nomor (1d), modifikasi arsitektur
tersebut untuk mendapatkan nilai akurasi optimal yang kalian dapatkan (kalian dapat
menambahkan atau mengurangi arsitektur tersebut, atau mengganti hyperparameter, atau
menggunakan tuning pada hyperparameter). Jelaskan alasan kalian untuk menggunakan
pendekatan yang kalian pilih terkait dengan tantangan dan dan kesulitan yang dihadapi dalam
data tabular tersebut (1c).

f. Lakukan evaluasi unjuk kerja kedua arsitektur di atas pada test set dengan
mencari nilai accuracy, precision, recall dan F1-Score. Dan berikan penjelasan mengenai hasilnya
dengan rinci.
