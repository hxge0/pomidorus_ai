# Pomidoruś AI

## 1. Informacje Ogólne

* **Nazwa modelu:** Pomidoruś AI


* **Typ modelu:** Klasyfikacja obrazów (Computer Vision / Convolutional Neural Networks)


* **Wersja:** 1.0 (Produkcyjna wersja oparta na Transfer Learningu)


* **Data wytrenowania:** Maj 2026 r.
* **Autor:** Krzysztof Falandysz


* **Architektury:** Prosta Sieć CNN / EfficientNetB0 (wariant pretrenowany na zbiorze ImageNet)

---

## 2. Dane Treningowe

* **Źródło danych:** Publiczny zbiór danych [Kaggle Tomato Leaf Dataset](https://www.kaggle.com/datasets/ashishmotwani/tomato) pobierany dynamicznie przez API Key.


* **Skala zbioru:** Ponad **25 000 zdjęć**.


* **Struktura klas:** 11 kategorii – w tym 10 specyficznych jednostek chorobowych oraz klasa reprezentująca liście zdrowe. Charakterystyka zdjęć obejmuje zarówno warunki laboratoryjne, jak i tzw. zdjęcia *in-the-wild* (naturalne warunki polowe).


* **Podział Zestawu Walidacyjnego:** Pierwotnie dataset nie posiadał setu testowego do późniejszej ewaluacji, dlatego postanowiłem podzielić go na pół. W pierwszej wersji na tym etapie popełniłem 2 duże błedy - nie przemieszałem danych w data secie (`shuffle=False`), przez co bazowy zbiór walidacyjny podzielił się w idealnej połowie na zbiór testowy i walidacyjny. Przez to Model ucząc się walidował wyniki jedynie na 4 klasach, a na macierzy pomyłek po ewaluacji modelu nie było tych czterech klas widać widać wcale. 

```python
TRAIN_DIR = "tomato/train"
VALID_DIR = "tomato/valid"

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

base_val_ds_val = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.5,
    subset="training",
    seed=42
)

base_val_ds_test = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.5,
    subset="validation",
    seed=42
)
```

---

## 3. Architektura i Strategia Treningowa

W procesie badawczym przetestowano dwa podejścia:

### Podejście A: Autorska sieć CNN (Własna architektura od zera)

Klasyczna sieć konwolucyjna składająca się z:

* 3 bloków `Conv2D` o rosnącej liczbie filtrów (32 → 64 → 128) z funkcją aktywacji **ReLU** do ekstrakcji krawędzi.


* Warstw `MaxPooling` po każdym bloku zmniejszających przestrzeń cech.


* Klasyfikatora: warstwy spłaszczającej `Flatten()` oraz warstwy w pełni połączonej `Dense` (128 neuronów).

```python

def build_tomato_cnn(input_shape=(128, 128, 3), num_classes=11):
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # Data Augmentation and Normalization
        data_augmentation,
        layers.Rescaling(1./255),

        # 1st Convolutional Layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 2nd Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 3rd Convolutional Layer
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten to 1 Vector
        layers.Flatten(),

        # Fully Connected Layer
        layers.Dense(128, activation='relu'),

        # Dropout
        layers.Dropout(0.5),

        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

```

<img width="350" height="180" alt="Metrics (3)" src="https://github.com/user-attachments/assets/bd112f0e-570c-4658-aa0e-4e45f5a2c699" />

<img width="214" height="180" alt="ConfMatrix" src="https://github.com/user-attachments/assets/fa238323-4463-416a-87df-441d3870eaba" />

<img width="564" height="200" alt="TrainingCharts" src="https://github.com/user-attachments/assets/d0c75788-377f-4c6b-af67-c6611b4158ce" />


### Podejście B: EfficientNetB0 + Transfer Learning (Wybrany model produkcyjny)

Zastosowanie zaawansowanej architektury z dwufazową strategią treningu:

```python

def build_model(num_classes, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    base_model.trainable = False

    x = layers.Rescaling(1./255)(base_model.output)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs), base_model

```

* **Faza 1 (Head Only):** Zamrożenie wag całej sieci bazowej i trenowanie wyłącznie nowo dodanych, wierzchnich warstw klasyfikacyjnych w celu stabilizacji wag wyjściowych.

```python

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Phase 1 compile

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Phase 1: training head only (base frozen)")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping, reduce_lr]
)

```

* **Faza 2 (Fine-tuning top-30):** Odmrożenie sieci bazowej i ponowne zamrożenie wszystkich warstw z wyjątkiem **ostatnich 30 warstw**. Dokładne dostrojenie najwyższych filtrów sieci do specyfiki detekcji chorób pomidora.

```python

base_model.trainable = True

total_layers = len(base_model.layers)
fine_tune_from = total_layers - 30

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Fine-tuning {trainable_count} of {total_layers} base layers")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping_phase2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

print("\nPhase 2: fine-tuning top 30 layers")
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stopping_phase2, reduce_lr]
)

```


<img width="350" height="180" alt="Metrics (1)" src="https://github.com/user-attachments/assets/3c4e0ff4-27e6-401c-bb99-bfbb02a694b8" />

<img width="214" height="180" alt="ConfusionMatrix" src="https://github.com/user-attachments/assets/765879dd-094e-400c-99a7-8a0ba9e49280" />

<img width="564" height="200" alt="TrainingChart" src="https://github.com/user-attachments/assets/b6724986-e37c-463e-ac7b-9b07f59e7e63" />


---

## 4. Wyniki Ewaluacji i Porównanie Modeli

Zastosowanie Transfer Learningu i architektury EfficientNetB0 pozwoliło na skokowy wzrost precyzji w stosunku do klasycznego podejścia CNN:

| Model | Ogólne Accuracy | Średnia F1-Score (Macro Avg) | Charakterystyka i zachowanie |
| --- | --- | --- | --- |
| **Własna sieć CNN (od zera)** | 83.9% | 83.7% | **Główny problem:** Poważne trudności z rozróżnianiem klasy *Late_blight* (precyzja tylko 67.2%). Duża liczba fałszywych alarmów. |
| **EfficientNetB0 (Fine-tuned)** | **97.2%** | **97.3%** | **Wybitna stabilność:** Dla niemal każdej choroby wskaźnik F1-score przekracza 95%. Najniższa odnotowana wartość dla pojedynczej klasy to wciąż wysokie 94.9%. |


---

## 5. Przeznaczenie i Zastosowanie

* **Główne zastosowanie:** Automatyczna, inteligentna klasyfikacja chorób liści pomidora na podstawie zdjęć aparatów fotograficznych. Targetowany jako moduł diagnostyczny czasu rzeczywistego (real-time).


* **Sugerowani odbiorcy:** Rolnicy, producenci żywności oraz twórcy aplikacji mobilnych dla ogrodnictwa (w celu minimalizacji globalnych strat plonów i redukcji kosztów eksperckich).


* **Formaty wdrożeniowe:** Model zapisany w natywnym formacie `.keras`, przygotowany do konwersji do formatu `.tflite` w celu lekkiego uruchamiania bezpośrednio na urządzeniach mobilnych w trybie **Offline**.


* **Ograniczenia (Out-of-scope):** Model został wytrenowany wyłącznie na liściach pomidora (11 kategorii). Nie należy stosować go do diagnozy innych gatunków roślin.
