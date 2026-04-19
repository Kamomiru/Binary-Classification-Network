# Binary Classification with PyTorch (Implicit Curves)

This personal project demonstrates how to train a small neural network using **PyTorch** to learn **decision boundaries** for binary classification problems defined by implicit mathematical curves.

The model learns to classify points in 2D space based on whether they lie **inside or outside** a given curve.

---

## 📌 Problem Overview

We define binary classification using implicit functions:

* Points where the function is **> 0 → Class A**
* Points where the function is **< 0 → Class B**

### Example Curves

#### Curve 1

$$
(x - 6)^3 + y^2 - 8 = 0
$$

#### Curve 2

$$
3x \sin(y) + 3y \cos(x) = 0
$$

---

## 🧠 Model Setup

To study the effects of different activation functions and optimizers I implemented following configurations.

| Optimizer | Activation |
| --------- | ---------- |
| SGD       | Sigmoid    |
| Adam      | Sigmoid    |
| SGD       | ReLU       |
| Adam      | ReLU       |

Each model is trained on the same dataset to compare performance.

---

## Loss Function
The networks have been trained using Binary Cross-Entropy Loss for logit outputs. So to aquire a prediction from the model it is necessary to apply the sigmoid function to the output.

$$
\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]
$$

---

## 📦 Model Architecture
Input layer: X and Y coordinates

Hidden layers:

Layer 1: 2 → 8

Layer 2: 8 → 8

Layer 3: 8 → 4

Output layer: 4 → 1 Logit output

---

## 📊 Results

### 🔹 Curve 1

#### Decision Boundary

The trained models approximate the true decision boundary of the implicit function:

<img width="1920" height="1015" alt="Decision Boundary Curve 1" src="https://github.com/user-attachments/assets/9b399ede-41e4-4532-9fb8-6ecc5c2e5c1a" />

#### Loss Curve

The following plot shows how training loss evolves over time for different model configurations:

<img width="1920" height="1015" alt="Loss Curve 1" src="https://github.com/user-attachments/assets/f2faad1d-2fa4-40b2-b8be-ceca7ccf6df5" />

#### Conclusion

It is clearly visible how different activation functions affect the model's approximations:

* **ReLU models** tend to produce more linear decision boundaries
* **Sigmoid models** produce smoother transitions

For such simple ground truths the sigmoid function is a feasable option but for more complex functions ReLU activation is significantly more performant. This can be observed in the second example.

The ground truth boundary is also plotted for comparison.

---

### 🔹 Curve 2

#### Decision Boundary

<img width="1920" height="1015" alt="Decision Boundary Curve 2" src="https://github.com/user-attachments/assets/43c52ac1-6795-401f-8603-8ab3c85fbaf2" />

#### Loss Curve

<img width="1920" height="1015" alt="Loss Curve 2" src="https://github.com/user-attachments/assets/36c1706d-4778-4293-a5d1-68db2f70bbd5" />

#### Conclusion

In the second example you can clearly see how the models are hitting thier limits.
SGD Sigmoid actually fails to learn at all and the other models accuracy has also decreased significally. 
As pointed out before the ReLU activations are now performing better than the sigmoid activations.

Of course these small examples would all benefit from adjusting hyperparameters.

---

## 🛠️ How to Train and Plot

You can easily experiment with different models, optimizers, and curves by modifying a few key files.

---

### ⚙️ Configuration Options

- **Model architecture**
  - Modify `model.py` to change layers, activations, or overall structure

- **Optimizer settings**
  - Adjust learning rate, momentum, or optimizer type in `main.py`

- **Dataset / curves**
  - Add or modify implicit functions in `DataHandler.py`

---

### 🚀 Running

#### 1. Train the model

Run training via:

```bash
python main.py
```
To switch between different curves, update the dataset definition in main.py:

```python
training_set = DataSet(40000, curve2_imp)
```
You can replace curve2_imp with any other function defined in DataHandler.py or wherever else you want

2. Visualize results

After training you can generate the same plots as above using:
```python
plot_results(curve2_imp)
```
the passed function should be the corresponding ground truths implicit function




## 🧩 Project Structure

```bash
.
├── model.py          # Neural network definition
├── DataHandler.py    # Dataset generation & curve definitions
├── training.py       # Training loop + accuracy computation
├── plotting.py       # Visualization of results
├── logger.py         # Saving/loading training data
├── data/             # Saved model weights
└── main.py           # Training entry point
```

## Disclaimer
I created this readme.md with the help of AI, but all content is authored by myself.


