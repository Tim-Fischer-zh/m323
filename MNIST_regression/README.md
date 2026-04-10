# M323 — Multiclass Classification für MNIST

Funktionale Programmierung mit F#: Ein neuronales Netz (MLP) von Grund auf implementiert.

Der Formelstandard kommt von dem [Stanford Online Machine Learning Kurs](https://www.coursera.org/specializations/machine-learning-introduction) (Andrew Ng).

> **Bemerkung zu KI:**
> Künstliche Intelligenz **wird** verwendet für: **Mathematik, Dokumentierung und Debugging**.
> Künstliche Intelligenz **wird nicht** verwendet für: **Code Generierung**.

---

## Inhaltsverzeichnis

1. [Setup](#1-setup)
2. [Features](#2-features)
3. [FP-Design-Entscheidungen](#3-fp-design-entscheidungen)
4. [Architektur](#4-architektur)
5. [Testing & Correctness](#5-testing--correctness)
6. [Mathematische Grundlagen](#6-mathematische-grundlagen)
7. [Notation](#7-notation)

---

## 1. Setup

### Voraussetzungen

- .NET 10
- NuGet Package: `FSharp.Data`

### Projektstruktur

```
MNIST_regression/
├── Program.fs              # Gesamter Code
├── data/
│   ├── mnist_train.csv     # 60'000 Trainingsbilder
│   └── mnist_test.csv      # 10'000 Testbilder
├── MNIST_regression.fsproj
└── README.md
```

### Starten

```bash
dotnet run
```

### Daten

Die MNIST-Daten liegen als CSV im Ordner `data/`. Jede Zeile enthält 785 Spalten: Spalte 0 ist das Label (0–9), Spalten 1–784 sind die Pixelwerte (0–255) eines 28×28 Bildes.

---

## 2. Features

- **Daten laden und normalisieren** — CSV einlesen mit `FSharp.Data.CsvFile`, Pixelwerte auf $[0, 1]$ skalieren
- **One-Hot Encoding** — Labels von Integer zu Vektor transformieren
- **He-Initialization** — Zufällige Startgewichte über Box-Muller-Transformation (Normalverteilung)
- **Forward Pass** — Matrix-Vektor-Multiplikation, ReLU-Activation, numerisch stabile Softmax
- **Cross-Entropy Loss** — Messung der Vorhersagequalität
- **Backpropagation** — Outer Product, transponierte Matrix-Vektor-Multiplikation, ReLU-Ableitung, Hadamard-Produkt
- **Parameter-Update** — Stochastic Gradient Descent
- **Rekursiver Training Loop** — `Array.fold` über Daten, rekursive Epochen-Schleife
- **Prediction und Accuracy** — Evaluation auf separaten Testdaten

---

## 3. FP-Design-Entscheidungen

### Immutability

Kein `let mutable` im gesamten Trainings- und Inferenz-Code. Jeder Trainingsschritt erzeugt **neue** Weights statt die bestehenden zu verändern:

```fsharp
// trainStep gibt neue Parameter zurück, alte bleiben unverändert
let trainstep (W1, b1, W2, b2) (label, pixels) =
    // ... Forward, Backward, Update ...
    W1New, b1New, W2New, b2New
```

**Bekannte Einschränkung:** Das globale `Random`-Objekt (`rnd`) ist mutable. Es wird ausschliesslich bei der einmaligen Weight-Initialization verwendet und hat keinen Einfluss auf den Trainingsprozess.

### Higher-Order Functions

Durchgehend verwendet statt imperativer Schleifen:

| HOF | Verwendung |
|-----|------------|
| `Array.init` | Weight-Initialization, Matrix-Vektor-Multiplikation, Outer Product |
| `Array.map` | Normalisierung, ReLU, Softmax, Skalierung |
| `Array.map2` | Vektor-Addition, Subtraktion, Hadamard-Produkt |
| `Array.sum` | Dot Product, Softmax-Nenner |
| `Array.filter` | Accuracy-Berechnung |
| `Array.fold` | Training Loop (ein kompletter Durchlauf über alle Daten) |
| `Array.findIndex` | Loss-Berechnung, argmax für Prediction |
| `Seq.map` | Daten-Pipeline (CSV-Zeilen transformieren) |

### Function Composition

Der Forward Pass ist eine Pipeline von Transformationen mit dem Pipe-Operator `|>`:

```fsharp
let z1 = matVecMul W1 pixels |> vecAdd b1
let a1 = relu z1
let z2 = matVecMul W2 a1 |> vecAdd b2
let yHat = softmax z2
```

Die Daten-Pipeline beim Einlesen ist ebenfalls als Komposition aufgebaut:

```fsharp
file.Rows
|> Seq.map (fun row -> ...)
|> Seq.toArray
```

### Recursion und Pattern Matching

Die Training-Schleife ist rekursiv statt iterativ implementiert, mit Pattern Matching auf den Epoch-Zähler:

```fsharp
let rec train (W1, b1, W2, b2) epoch =
    match epoch with
    | 0 -> (W1, b1, W2, b2)
    | n ->
        let W1New, b1New, W2New, b2New = Array.fold trainstep (W1, b1, W2, b2) data
        train (W1New, b1New, W2New, b2New) (n - 1)
```

Tuple-Destructuring (ebenfalls Pattern Matching) wird durchgehend verwendet:

```fsharp
let label, pixels = data.[0]
```

### Type Safety

Alle Funktionen haben explizite oder inferierte Typsignaturen. Die zwei zentralen Datenstrukturen sind klar getrennt:

| Typ | Verwendung |
|-----|------------|
| `float[,]` | Weight-Matrizen ($W^{[1]}$, $W^{[2]}$) |
| `float array` | Vektoren (Inputs, Biases, Activations, Gradienten) |
| `(float array * float array)` | Datenpunkte (Label, Pixel) als Tuple |

Mathematische Operationen sind typensicher — `matVecMul` akzeptiert nur `float[,]` und `float array`, nicht umgekehrt.

---

## 4. Architektur

```
Input [784] → Hidden [128, ReLU] → Output [10, Softmax]
```

Lernbare Parameter: ~101'770 (Weights + Biases)

| Parameter | Dimension | Anzahl |
|-----------|-----------|--------|
| $W^{[1]}$ | $128 \times 784$ | 100'352 |
| $b^{[1]}$ | $128 \times 1$ | 128 |
| $W^{[2]}$ | $10 \times 128$ | 1'280 |
| $b^{[2]}$ | $10 \times 1$ | 10 |

### Hyperparameter

| Parameter | Wert |
|-----------|------|
| Hidden Neurons | 128 |
| Learning Rate $\alpha$ | 0.001 |
| Epochen | 3 |
| Initialization | He ($\sqrt{2/n_{\text{in}}}$) |
| Batch-Grösse | 1 (Stochastic Gradient Descent) |

### Implementierte Operationen

| Funktion | Operation | Signatur |
|----------|-----------|----------|
| `matVecMul` | $W \cdot x$ | `float[,] → float array → float array` |
| `matTransVecMul` | $W^T \cdot x$ | `float[,] → float array → float array` |
| `vecAdd` | $a + b$ | `float array → float array → float array` |
| `vecSub` | $a - b$ | `float array → float array → float array` |
| `vecMul` | $a \odot b$ | `float array → float array → float array` |
| `outerProduct` | $a \cdot b^T$ | `float array → float array → float[,]` |
| `matSub` | $A - B$ | `float[,] → float[,] → float[,]` |
| `matScale` | $\alpha \cdot M$ | `float → float[,] → float[,]` |
| `relu` | $\max(0, z)$ | `float array → float array` |
| `reluDerivative` | $g'(z)$ | `float array → float array` |
| `softmax` | Softmax mit num. Stabilität | `float array → float array` |
| `loss` | $-\log(\hat{y}_d)$ | `float array → float array → float` |
| `onehot` | Label → One-Hot-Vektor | `int → float array` |

> **Notation:** $\hat{y} = a^{[2]}$ — der Output des letzten Layers ist gleichzeitig die Vorhersage.

---

## 5. Testing & Correctness

### Loss-Verlauf

Der Loss sinkt konsistent über die Epochen, was korrektes Lernen bestätigt:

```
Epoch 3  Loss: 0.203189
Epoch 2  Loss: 0.103439
Epoch 1  Loss: 0.009811
```

### Accuracy

Nach 3 Epochen Training auf 60'000 Bildern, evaluiert auf 10'000 separaten Testbildern:

```
Accuracy: 94.76%
```

Zum Vergleich: Zufälliges Raten ergäbe ~10% (10 Klassen). Lineare Softmax-Regression ohne Hidden Layer erreicht ~92%.

### Correctness-Argumente

- **Softmax:** Output summiert zu 1.0, alle Werte zwischen 0 und 1 — verifiziert durch `Array.sum yHat`
- **One-Hot:** Genau eine 1.0 pro Vektor an der korrekten Index-Position — manuell geprüft gegen CSV-Labels
- **Normalisierung:** Pixelwerte liegen nach Division durch 255 im Bereich $[0, 1]$
- **Dimensionen:** Alle Matrix-Vektor-Operationen produzieren korrekte Ausgabe-Dimensionen ($128 \times 784 \cdot 784 \rightarrow 128$, $10 \times 128 \cdot 128 \rightarrow 10$)
- **Gradient Descent:** Loss sinkt monoton über Epochen — Gradienten zeigen in die richtige Richtung

### Bekannte Einschränkungen

- **Performance:** Kein Mini-Batching, kein Parallelismus — 3 Epochen dauern ~10 Minuten
- **Mutable State:** `let rnd = new Random()` ist ein globales mutable Objekt (nur bei Initialization verwendet)
- **Keine externe ML-Library:** Alle Operationen (Matrix-Multiplikation, Softmax, Backprop) sind von Hand implementiert

---

## 6. Mathematische Grundlagen

### Datenaufbereitung

**Normalisierung** — Pixelwerte von $[0, 255]$ auf $[0, 1]$ skalieren:

$$x_i = \frac{\text{pixel}_i}{255}$$

**One-Hot Encoding** — Label $d \in \{0, \ldots, 9\}$ wird zu Vektor $y \in \mathbb{R}^{10}$:

$$y_k = \begin{cases} 1 & k = d \\ 0 & k \neq d \end{cases}$$

### Weight Initialization (He)

$$W^{[l]}_{ij} \sim \mathcal{N}\left(0, \; \sqrt{\frac{2}{n^{[l-1]}}}\right)$$

$$b^{[l]} = \vec{0}$$

Jeder Gewichtswert wird zufällig aus einer Normalverteilung gezogen (Box-Muller-Transformation). Mittelwert $0$, Standardabweichung $\sqrt{2 / n_{\text{in}}}$ wobei $n_{\text{in}}$ die Anzahl Inputs des Layers ist. Biases starten bei Null.

---

### Forward Pass

**Layer 1 — Hidden (ReLU):**

$$z^{[1]} = W^{[1]} \cdot \mathbf{x} + b^{[1]}$$

$$a^{[1]} = g(z^{[1]}) = \max(0, \; z^{[1]})$$

**Layer 2 — Output (Softmax):**

$$z^{[2]} = W^{[2]} \cdot a^{[1]} + b^{[2]}$$

$$\hat{y}_j = \frac{e^{z^{[2]}_j - \max(z^{[2]})}}{\displaystyle\sum_{k=1}^{N} e^{z^{[2]}_k - \max(z^{[2]})}}$$

Der Abzug von $\max(z^{[2]})$ dient der numerischen Stabilität und verändert das Ergebnis mathematisch nicht.

---

### Activation Functions

**ReLU** — setzt negative Werte auf Null:

$$g(z) = \max(0, \; z)$$

$$g'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

**Softmax** — wandelt rohe Werte in eine Wahrscheinlichkeitsverteilung um ($\sum \hat{y}_k = 1$):

$$a_j = \frac{e^{z_j}}{\displaystyle\sum_{k=1}^{N} e^{z_k}}$$

---

### Loss Function

**Cross-Entropy Loss** für ein einzelnes Sample:

$$L(\hat{y}, y) = -\sum_{j=1}^{N} \mathbf{1}\{y = j\} \log(a_j) = -\log(\hat{y}_d)$$

Vereinfachung gilt weil $y$ ein One-Hot Vektor ist — nur der Term mit $j = d$ (korrekte Klasse) überlebt.

### Cost Function (über alle Samples)

$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{N} \mathbf{1}\{y^{(i)} = j\} \log \frac{e^{z^{(i)}_j}}{\sum_{k=1}^{N} e^{z^{(i)}_k}}$$

---

### Backward Pass (Backpropagation)

**Output-Layer Fehlersignal** (Softmax + Cross-Entropy kombiniert):

$$\delta^{[2]} = \hat{y} - y$$

**Gradienten für Layer 2:**

$$\frac{\partial J}{\partial W^{[2]}} = \delta^{[2]} \cdot (a^{[1]})^T$$

$$\frac{\partial J}{\partial b^{[2]}} = \delta^{[2]}$$

**Hidden-Layer Fehlersignal** (rückwärts propagiert):

$$\delta^{[1]} = (W^{[2]})^T \cdot \delta^{[2]} \;\odot\; g'(z^{[1]})$$

Wobei $\odot$ die elementweise Multiplikation (Hadamard-Produkt) bezeichnet.

**Gradienten für Layer 1:**

$$\frac{\partial J}{\partial W^{[1]}} = \delta^{[1]} \cdot \mathbf{x}^T$$

$$\frac{\partial J}{\partial b^{[1]}} = \delta^{[1]}$$

---

### Parameter Update (Gradient Descent)

$$W^{[l]} \leftarrow W^{[l]} - \alpha \; \frac{\partial J}{\partial W^{[l]}}$$

$$b^{[l]} \leftarrow b^{[l]} - \alpha \; \frac{\partial J}{\partial b^{[l]}}$$

Wobei $\alpha$ die Learning Rate ist.

---

### Evaluation

$$\text{Prediction} = \arg\max(\hat{y})$$

Die vorhergesagte Klasse ist der Index mit der höchsten Wahrscheinlichkeit im Output-Vektor.

---

### Training Loop (Pseudocode)

```
Initialisiere W[1], b[1], W[2], b[2]

Für jede Epoch:
    Für jedes Trainingsbeispiel (x, y):

        // Forward
        z[1] = W[1] · x + b[1]
        a[1] = max(0, z[1])
        z[2] = W[2] · a[1] + b[2]
        ŷ    = Softmax(z[2])

        // Loss
        L = -log(ŷ_d)

        // Backward
        δ[2]     = ŷ - y
        ∂J/∂W[2] = δ[2] · a[1]ᵀ
        ∂J/∂b[2] = δ[2]
        δ[1]     = W[2]ᵀ · δ[2] ⊙ g'(z[1])
        ∂J/∂W[1] = δ[1] · xᵀ
        ∂J/∂b[1] = δ[1]

        // Update
        W[l] ← W[l] - α · ∂J/∂W[l]
        b[l] ← b[l] - α · ∂J/∂b[l]

    Accuracy auf Testdaten messen
```

---

## 7. Notation

| Symbol | Bedeutung |
|--------|-----------|
| $\mathbf{x}$ | Input-Vektor (784 normalisierte Pixelwerte) |
| $y$ | One-Hot-Vektor des Labels |
| $\hat{y}$ | Vorhergesagte Wahrscheinlichkeitsverteilung ($= a^{[2]}$) |
| $W^{[l]}$ | Weight-Matrix von Layer $l$ |
| $b^{[l]}$ | Bias-Vektor von Layer $l$ |
| $z^{[l]}$ | Pre-Activation (vor Activation Function) |
| $a^{[l]}$ | Activation (nach Activation Function) |
| $g(z)$ | Activation Function (ReLU oder Softmax) |
| $g'(z)$ | Ableitung der Activation Function |
| $\delta^{[l]}$ | Fehlersignal von Layer $l$ |
| $\alpha$ | Learning Rate |
| $\odot$ | Elementweise Multiplikation (Hadamard-Produkt) |
| $L$ | Loss (einzelnes Sample) |
| $J$ | Cost Function (über alle Samples) |
| $m$ | Anzahl Trainingsbeispiele |
| $N$ | Anzahl Output-Klassen (10) |
| $d$ | Korrekte Klasse (0–9) |
| $\mathbf{1}\{y = j\}$ | Indikator-Funktion: 1 wenn $y = j$, sonst 0 |