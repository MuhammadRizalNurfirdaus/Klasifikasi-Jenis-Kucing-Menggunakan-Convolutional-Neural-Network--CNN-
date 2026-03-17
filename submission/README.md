# Klasifikasi Jenis Kucing Menggunakan CNN

Proyek klasifikasi gambar kucing menggunakan **Convolutional Neural Network (CNN)** dengan TensorFlow/Keras.

## Kelas Dataset

| Kelas | Deskripsi |
|:------|:----------|
| Belang Tiga | Kucing tiga warna (Tricolor/Tortoiseshell) |
| Hitam | Kucing berwarna hitam |
| Kampung | Kucing kampung/lokal Indonesia |

## Struktur Direktori

```
submission/
├── tfjs_model/
│   ├── group1-shard1of1.bin
│   └── model.json
├── tflite/
│   ├── model.tflite
│   └── label.txt
├── saved_model/
│   ├── saved_model.pb
│   └── variables/
├── notebook.ipynb
├── README.md
└── requirements.txt
```

## Cara Menjalankan

1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan `notebook.ipynb` dari awal hingga akhir.

3. Pastikan Kaggle API credentials sudah dikonfigurasi (`~/.kaggle/kaggle.json`).

## Kriteria yang Dipenuhi

- ✅ Dataset minimal 1000 gambar
- ✅ Split Train (80%) / Validation (10%) / Test (10%)
- ✅ Model Sequential + Conv2D + MaxPooling2D
- ✅ Akurasi ≥ 85% pada train & test
- ✅ Plot akurasi dan loss
- ✅ Disimpan dalam SavedModel, TF-Lite, dan TFJS
- ✅ Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- ✅ 3 kelas
- ✅ Demo inferensi TF-Lite dengan output
