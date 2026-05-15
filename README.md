# Tubes2 IF3270 Pembelajaran Mesin

Repository ini berisi implementasi Tugas Besar 2 IF3270 Pembelajaran Mesin untuk dua bagian utama:

- CNN untuk klasifikasi gambar Intel Image Classification.
- RNN dan LSTM untuk image captioning Flickr8k.

Implementasi reusable berada di `src/`, sedangkan notebook digunakan untuk training, pengujian, visualisasi, dan analisis hasil.

## Struktur Repository

```text
src/common/       Utilitas untuk image loading dan feature extraction
src/cnn/          Implementasi CNN, layer from scratch, training helper, dan evaluasi
src/captioning/   Pipeline Flickr8k captioning, decoder Keras, dan decoder from scratch
notebook/         Notebook eksperimen dan analisis
doc/              Spesifikasi tugas dan laporan
artifacts/        Output eksperimen, model, history, evaluasi, dan grafik (gitignored)
data/             Dataset raw dan processed (gitignored)
```

## Setup Environment

Gunakan Python `>=3.11`, lalu jalankan dari root repository:

```powershell
pip install -r requirements.txt
pip install -e .
```

## Dataset Layout

Siapkan dataset lokal dengan struktur berikut.

Intel Image Classification:

```text
data/raw/intel/
```

Flickr8k:

```text
data/raw/flickr8k/Images/*.jpg
data/raw/flickr8k/captions.txt
```

Output hasil preprocessing, feature extraction, model weights, history training, evaluasi, dan grafik disimpan di `artifacts/`. Folder `data/` dan `artifacts/` berisi file besar dan sengaja diabaikan oleh Git.

## CNN Workflow

Notebook utama untuk bagian CNN:

```text
notebook/cnn-train.ipynb
notebook/cnn-forward.ipynb
```

Gunakan `cnn-train.ipynb` untuk training dan eksperimen model CNN. Gunakan `cnn-forward.ipynb` untuk pengujian forward propagation from scratch dan perbandingan hasil terhadap Keras.

Implementasi CNN berada di:

```text
src/cnn/core/       Abstraksi model dan operasi inti
src/cnn/nn/         Layer, aktivasi, metric, dan adapter Keras
src/cnn/data/       Dataset dan image loading helper
src/cnn/training/   Helper training, eksperimen, dan serialization
```

## RNN/LSTM Captioning Workflow

Urutan utama untuk mereproduksi pipeline RNN/LSTM dari artifact kosong:

```powershell
python -m captioning.extract_inception_features
python -m captioning.preprocess_captions
python -m captioning.build_teacher_forcing
python -m captioning.train_decoders
python -m captioning.fafo.training_results --eval-table
python -m captioning.fafo.scratch_forward_inference --experiment-id lstm_layers1_hidden128 --count 1000
python -m captioning.fafo.scratch_forward_inference --experiment-id rnn_layers2_hidden128 --count 1000
```

Perintah tersebut melakukan feature extraction InceptionV3, preprocessing caption, pembuatan teacher-forcing arrays, training 12 variasi pre-inject, evaluasi BLEU-4/METEOR/waktu inference, dan validasi Keras vs NumPy scratch pada model terpilih.

Eksperimen bonus dan analisis tambahan:

```powershell
python -m captioning.train_init_inject_decoders
python -m captioning.fafo.training_results --summary-path artifacts/captioning/experiments/initinject_decoder_training_summary.json --eval-table
python -m captioning.fafo.beam_search_comparison --experiment-id lstm_layers1_hidden128 --count 200 --beam-width 3
python -m captioning.fafo.beam_search_comparison --experiment-id rnn_layers2_hidden128 --count 200 --beam-width 3
python -m captioning.fafo.max_caption_length_experiment --experiment-id lstm_layers1_hidden128 --backend scratch --count 200 --max-steps 15,20,30,37
python -m captioning.fafo.comparison_examples --rnn-experiment-id rnn_layers2_hidden128 --lstm-experiment-id lstm_layers1_hidden128 --count 1000 --output-count 30
```

Output utama dari pipeline disimpan di `artifacts/captioning/`, termasuk feature InceptionV3, vocabulary, teacher-forcing arrays, weights decoder, history training, tabel evaluasi, dan hasil perbandingan Keras vs scratch.

## Notebook Experiment

Notebook analisis utama dan experiment untuk bagian captioning rnn dan lstm:

```text
notebook/rnn_lstm_captioning_experiment.ipynb
```

Notebook tersebut membaca artifact yang sudah ada dan menyajikan:

- Ringkasan dataset, feature extraction, dan preprocessing.
- Evaluasi 12 variasi pre-inject RNN/LSTM.
- Pengaruh jumlah layer recurrent dan hidden state.
- Kurva training loss dan validation loss.
- Perbandingan RNN vs LSTM.
- Perbandingan Keras vs NumPy scratch, termasuk execution time.
- Pengaruh panjang maksimum caption.
- Analisis kualitatif high/medium/low examples.
- Bonus init-inject dan beam search.
- Sintesis temuan utama.

## Artifact Penting RNN/LSTM

Beberapa artifact utama yang digunakan notebook analisis:

```text
artifacts/captioning/features/inception_v3_flickr8k_features.npy
artifacts/captioning/preprocessed/caption_preprocessing_metadata.json
artifacts/captioning/teacher_forcing/teacher_forcing_metadata.json
artifacts/captioning/experiments/decoder_training_summary.json
artifacts/captioning/fafo/test_evaluation_full_images.csv
artifacts/captioning/fafo/lstm_layers1_hidden128_test_keras_vs_scratch_500_images.csv
artifacts/captioning/fafo/rnn_layers2_hidden128_test_keras_vs_scratch_500_images.csv
artifacts/captioning/fafo/initinject_test_evaluation_full_images.csv
artifacts/captioning/fafo/comparison_20_examples.csv
```

## Pembagian Tugas

```text
Nama/NIM 1: CNN / RNN-LSTM / laporan / integrasi
Nama/NIM 2: CNN / RNN-LSTM / laporan / integrasi
Nama/NIM 3: CNN / RNN-LSTM / laporan / integrasi
```

## Task List Spesifikasi Wajib

| Task | Dikerjakan | Tidak dikerjakan |
|---|---|---|
| [CNN] Utility image processing dan feature extraction untuk dataset gambar | [x] | [ ] |
| [CNN] Implementasi forward propagation CNN from scratch yang dapat load bobot Keras | [x] | [ ] |
| [CNN] Training model CNN Keras untuk klasifikasi Intel Image Classification | [x] | [ ] |
| [CNN] Eksperimen hyperparameter CNN sesuai spesifikasi: jumlah layer, jumlah filter, ukuran filter, dan jenis pooling | [x] | [ ] |
| [CNN] Evaluasi dan analisis CNN: macro F1-score, Keras vs scratch, shared vs non-shared, grafik loss, dan jumlah parameter | [x] | [ ] |
| [RNN/LSTM] Feature extraction Flickr8k dengan CNN encoder frozen dan preprocessing caption | [x] | [ ] |
| [RNN/LSTM] Implementasi decoder Keras pre-inject untuk SimpleRNN dan LSTM | [x] | [ ] |
| [RNN/LSTM] Training 12 variasi decoder: 6 SimpleRNN dan 6 LSTM | [x] | [ ] |
| [RNN/LSTM] Implementasi forward propagation decoder from scratch untuk RNN dan LSTM | [x] | [ ] |
| [RNN/LSTM] Pipeline image captioning from scratch dari raw image sampai caption | [x] | [ ] |
| [RNN/LSTM] Evaluasi dan analisis RNN/LSTM: BLEU-4, METEOR, waktu eksekusi, Keras vs scratch, RNN vs LSTM, grafik loss, contoh kualitatif, dan panjang maksimum caption | [x] | [ ] |

## Task List Spesifikasi Bonus

| Task | Dikerjakan | Tidak dikerjakan |
|---|---|---|
| [CNN] Visualisasi intermediate feature maps dari layer konvolusi | [x] | [ ] |
| [CNN] Grad-CAM untuk region gambar yang paling berpengaruh terhadap prediksi | [x] | [ ] |
| [RNN/LSTM] Implementasi image captioning init-inject sebagai alternatif pre-inject dan membandingkan hasilnya| [x] | [ ] |
| [RNN/LSTM] Implementasi beam search decoder dengan `k = 3` atau `k = 5` dan membandingkan hasilnya | [x] | [ ] |
| [Semua] Batch inference untuk seluruh forward propagation from scratch dengan `batch_size` | [x] | [ ] |
| [Semua] Backward propagation from scratch untuk seluruh layer yang digunakan | [ ] | [x] |