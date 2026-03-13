# M323 — Multiclass Classification für MNIST

Funktionale Programmierung mit F#: Ein neuronales Netz (MLP) von Grund auf implementiert.

Der Formelstandard kommt von dem [Stanford Online Machine Learning Kurs](https://www.coursera.org/specializations/machine-learning-introduction)

Bemerkung zu KI: 
> Künstliche Intelligenz **wird** verwendet für: **Mathematik, Dokumentierung und Debugging**. 
Künstliche Intelligenz **wird nicht** verwendet für: **Code Generierung**

## Architektur

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

> **Notation:** $\hat{y} = a^{[2]}$ — der Output des letzten Layers ist gleichzeitig die Vorhersage.

---

## Mathematische Grundlagen

### Datenaufbereitung

**Normalisierung** — Pixelwerte von $[0, 255]$ auf $[0, 1]$ skalieren:

$$x_i = \frac{\text{pixel}_i}{255}$$

**One-Hot Encoding** — Label $d \in \{1, \ldots, 10\}$ wird zu Vektor $y \in \mathbb{R}^{10}$:

$$y_k = \begin{cases} 1 & k = d \\ 0 & k \neq d \end{cases}$$

### Weight Initialization (He)

$$W^{[l]}_{ij} \sim \mathcal{N}\left(0, \; \sqrt{\frac{2}{n^{[l-1]}}}\right)$$

$$b^{[l]} = \vec{0}$$

Jeder Gewichtswert wird zufällig aus einer Normalverteilung gezogen. Mittelwert $0$, Standardabweichung $\sqrt{2 / n_{\text{in}}}$ wobei $n_{\text{in}}$ die Anzahl Inputs des Layers ist. Biases starten bei Null.

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


## Hyperparameter

| Parameter | Wert |
|-----------|------|
| Hidden Neurons | 128 |
| Learning Rate $\alpha$ | 0.01 – 0.1 |
| Epochen | 10 – 30 |
| Initialization | He ($\sqrt{2/n_{\text{in}}}$) |

---

## Notation

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
| $d$ | Korrekte Klasse |
| $\mathbf{1}\{y = j\}$ | Indikator-Funktion: 1 wenn $y = j$, sonst 0 |