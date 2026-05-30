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

## 6. Wnioski i Podsumowanie Projektu (Conclusions & Key Takeaways)

Na podstawie przeprowadzonych eksperymentów badawczych i procesu trenowania modeli "Pomidoruś AI" sformułowano następujące wnioski:

1. **Przewaga Transfer Learningu nad klasyczną architekturą:**

   Zastosowanie pretrenowanej sieci `EfficientNetB0` z dwufazowym dostrajaniem (Fine-tuning ostatnich 30 warstw) przyniosło skokowy wzrost skuteczności (ogólne Accuracy wzrosło z **83.9%** do aż **97.2%**). Dowodzi to, że w zadaniach klasyfikacji chorób roślin cechy wyekstrahowane na ogólnym zbiorze (ImageNet) stanowią doskonałą bazę do dalszej specjalizacji modelu.

---

2. **Rozwiązanie problemu fałszywych alarmów (Klasa Late_blight):**

   Autorska sieć CNN budowana od zera charakteryzowała się niską precyzją w wykrywaniu zarazy ziemniaczanej pomidora (*Late_blight*), osiągając wskaźnik na poziomie zaledwie **67.2%**. Model ten mylił cechy wizualne tej choroby z innymi kategoriami. Przejście na architekturę `EfficientNetB0` ustabilizowało proces predykcji, podnosząc wskaźniki F1-score dla wszystkich klas powyżej poziomu **94.9%**.

---

3. **Kluczowa rola symulacji warunków polowych (Augmentacja):**

   Zastosowanie pipeline'u transformacji geometrycznych i optycznych (modyfikacja jasności i kontrastu o 20%) pozwoliło modelowi na poprawną pracę ze zdjęciami typu *in-the-wild*. Dzięki temu model nie przeuczył się do idealnych, jednolitych warunków laboratoryjnych z bazy Kaggle, lecz zyskał odporność na zmienne oświetlenie występujące naturalnie na polach uprawnych.

---

4. **Wydajność potoku danych (Data Pipeline):**

   Zaimplementowanie mechanizmu `prefetch` wyeliminowało wąskie gardło (bottleneck) związane z ładowaniem obrazów w czasie rzeczywistym z dysku, co przełożyło się na maksymalne wykorzystanie mocy obliczeniowej procesora graficznego (GPU) i znaczące skrócenie czasu treningu.
   
---

5. **Ewolucja poprzez błędy (Lessons Learned):**

   Proces tworzenia projektu uwzględniał iteracyjne podejście (zgodnie z sekcją "faile projektowe"). Pierwsze nieudane próby z własną architekturą pozwoliły lepiej zrozumieć złożoność cech morfologicznych liści pomidora i wymusiły zmianę strategii na podejście z zaawansowanym Transfer Learningiem, co ostatecznie doprowadziło do osiągnięcia stabilności produkcyjnej.

---

6. **Wysoki potencjał wdrożeniowy brzegowego AI (Edge AI):**

   Wytrenowany model wykazuje pełną gotowość do konwersji do formatu `.tflite`. Niski narzut obliczeniowy architektury `EfficientNetB0` w połączeniu z możliwością działania **w 100% Offline** sprawia, że system jest idealnym rozwiązaniem do implementacji w docelowej aplikacji mobilnej dla rolników - gotowej do działania w miejscach o ograniczonym dostępie do sieci komórkowych.

---

7. **Znaczenie prawidłowego podziału zbioru danych (Data Leakage & Evaluation Bias):**

   Wczesne fazy projektu wykazały, jak krytyczna dla wiarygodności modelu jest poprawna implementacja podziału danych (*validation/test split*). Błąd w konfiguracji potoku danych doprowadził do sytuacji, w której zbiór walidacyjny zawierał próbki pochodzące jedynie z 4 klas, podczas gdy ostateczna ewaluacja została przeprowadzona na pozostałych kategoriach. W efekcie macierz pomyłek (*Confusion Matrix*) nie odzwierciedlała pełnego spektrum klas (całkowity brak pierwszych 4 klas). Ewaluacja modelu wyłącznie na części klas drastycznie zniekształca rzeczywisty obraz jego skuteczności. Może prowadzić do sytuacji, w której model wydaje się doskonale radzić sobie z problemem (wysokie ogólne Accuracy na uciętym zbiorze walidacyjnym), podczas gdy w warunkach produkcyjnych kompletnie zawiedzie przy podaniu obrazu z pominiętych kategorii. Prawidłowy podział musi zawsze gwarantować reprezentatywność (np. poprzez walidację stratyfikowaną – *Stratified Split*), aby każda klasa miała proporcjonalny udział w zbiorze uczącym, walidacyjnym i testowym.

<img width="563" height="485" alt="BadValTestSegmentationEffect" src="https://github.com/user-attachments/assets/d8e38a51-6ee8-461d-815d-b7d42fa5a350" />

---

8. **Konieczność stosowania mechanizmów zapisu stanu (Model Checkpointing):**

   Podczas realizacji projektu wystąpiła krytyczna sytuacja techniczna – utrata połączenia ze środowiskiem wykonawczym (chmurą obliczeniową) na ostatniej, 30. epoce treningu. Z powodu braku konfiguracji automatycznego zapisu, cały proces uczenia oraz wypracowane wagi modelu zostały utracone, co wygenerowało niepotrzebne koszty infrastruktury i stratę czasu. W procesie trenowania sieci neuronowych absolutnym standardem powinno być stosowanie tzw. *callbacków* zapisujących stan modelu (np. `ModelCheckpoint` w Keras/TensorFlow). Zapisywanie wag po każdej epoce (lub tylko wtedy, gdy model poprawia swój wynik na zbiorze walidacyjnym) zabezpiecza projekt przed awariami sprzętowymi, przerwami w dostawie prądu czy rozłączeniem sesji w chmurze (np. na Google Colab/Kaggle Notebooks).

<img width="821" height="615" alt="download" src="https://github.com/user-attachments/assets/51946af4-0686-4b2a-9724-a0563647a2df" />

---

9. **Zarządzanie budżetem obliczeniowym a przyrost efektywności (Wczesne zatrzymanie / Early Stopping):**

   Eksperymenty wykazały, że czekanie przez kolejne 15 epok na minimalną poprawę celności (*Accuracy*) na zbiorze walidacyjnym o symboliczną wartość (np. 1%) jest nieuzasadnione ekonomicznie i inżyniersko. Koszty zużycia energii elektrycznej oraz wynajmu mocy obliczeniowej GPU drastycznie przewyższają potencjalne korzyści z tak niewielkiego zysku dokładności. Proces trenowania AI wymaga balansowania między dokładnością a kosztem. Aby zoptymalizować ten proces, należy wdrażać mechanizm `EarlyStopping` z odpowiednio dobranym parametrem `patience` (np. zatrzymanie treningu, jeśli przez 5-7 epok strata walidacyjna nie maleje). Pozwala to na drastyczne skrócenie czasu pracy maszyn, chroni model przed przeuczeniem (*overfitting*) i optymalizuje budżet projektu.
