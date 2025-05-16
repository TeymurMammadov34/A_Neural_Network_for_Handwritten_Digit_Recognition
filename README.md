# 🧠 MNIST El Yazısı Rakam Tanıma (Artificial Neural Network)

Bu proje, Keras kütüphanesi kullanılarak el yazısı rakamların tanınması amacıyla yapay sinir ağı (ANN) modeli oluşturur ve eğitir. Kullanılan veri seti, geniş çapta bilinen **MNIST (Modified National Institute of Standards and Technology)** veri setidir.

## 🗂️ İçerik

- [Proje Hakkında](#ProjeHakkında)
- [Gereksinimler](#gereksinimler)
- [Veri Seti ve Ön İşleme](#veri-seti-ve-ön-işleme)
- [Model Mimarisi](#model-mimarisi)
- [Model Eğitimi](#model-eğitimi)
- [Model Değerlendirme ve Sonuçlar](#model-değerlendirme-ve-sonuçlar)
- [Model Kaydetme ve Yükleme](#model-kaydetme-ve-yükleme)
- [Sonuçlar ve Grafikler](#sonuçlar-ve-grafikler)

---

## 📌 Proje Hakkında

Bu proje, 28x28 boyutundaki gri tonlamalı el yazısı rakam görsellerini alır ve bir sinir ağı kullanarak 0-9 arasındaki rakamları tahmin etmeye çalışır. Modelin amacı, test seti üzerinde mümkün olan en yüksek doğruluğu elde etmektir.

---

## 🧰 Gereksinimler

Aşağıdaki Python kütüphaneleri gereklidir:

```bash
pip install tensorflow keras matplotlib
````

Kullanılan başlıca kütüphaneler:
- TensorFlow / Keras
- NumPy
- Matplotlib


## 📊 Veri Seti ve Ön İşleme
- MNIST veri seti keras.datasets üzerinden yüklenmiştir.
- Görseller 0-255 arası piksel değerlerinden 0-1 aralığına normalize edilmiştir.
- Giriş verileri 784 boyutlu (28x28) vektörlere dönüştürülmüştür.
- Etiketler one-hot encoding yöntemiyle dönüştürülmüştür.


## 🧠 Model Mimarisi
```bash
model = Sequential()

model.add(Dense(512, activation="relu", input_shape=(784,)))

model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))
```
- Katman 1: 512 nöron, ReLU aktivasyon
- Katman 2: 256 nöron, ReLU aktivasyon
- Çıkış Katmanı: 10 nöron, Softmax aktivasyon

## ⚙️ Model Derleme

```bash
  model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

## 🏋️ Model Eğitimi
```bash
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath="ann_best_model.keras", monitor="val_loss", save_best_only=True)

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)
```
- Eğitimde **EarlyStopping** ve **ModelCheckpoint** callback'leri kullanılmıştır.
- **validation_split=0.2**, **epochs=20**, **batch_size=64**

## 📈 Model Değerlendirme ve Sonuçlar
```bash
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.2f}")
```
#### 🔹 Test Doğruluğu: **%98+**
#### 🔹 Test Hatası: **0.07 civarı**

## 💾 Model Kaydetme ve Yükleme
```bash
model.save("final_mnist_ann_model.keras")
loaded_model = load_model("final_mnist_ann_model.keras")
```
Model başarıyla tekrar yüklenebilir ve test seti üzerinde yüksek doğrulukla tahmin yapabilir.

## 📊 Sonuçlar ve Grafikler

Aşağıdaki grafiklerde eğitim ve doğrulama sırasında modelin doğruluk ve kayıp değerleri gösterilmiştir:
- Eğitim ve Doğrulama Doğruluğu
- Eğitim ve Doğrulama Kaybı
```bash
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(history.history["accuracy"], "o-", label="Train Accuracy")
ax[0].plot(history.history["val_accuracy"], "s-", label="Validation Accuracy")
ax[0].set_title("Accuracy")
ax[0].legend()

ax[1].plot(history.history["loss"], "o-", label="Train Loss")
ax[1].plot(history.history["val_loss"], "s-", label="Validation Loss")
ax[1].set_title("Loss")
ax[1].legend()
plt.show()
```
## 📌 Notlar
- Model **adam** optimizasyon algoritmasıyla oldukça hızlı öğrenmiştir.

- MNIST gibi basit veri setlerinde ANN, yüksek başarı oranına ulaşabilir.

- Daha ileri düzey için CNN (Convolutional Neural Network) modelleri de incelenebilir.

## 👨‍💻 Geliştirici

**Teymur Mammadov** 

📧 İletişim: [timurmammadov34@gmail.com]


## 📁 Lisans
Bu proje MIT lisansı ile lisanslanmıştır. Detaylar için **LICENSE** dosyasını inceleyin.
