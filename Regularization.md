- L2 Regularization
- Dropout 
	- Spread out weights
	- Adaptive L2 norm
	- Inverted Dropout  - Divide the activation by keep.prob so as to retain the expected value of activation at that layer
		- Dropout is not used at test time 
		- This is the reason we divide by keep prob so that expected value of the activation don't change
	- Downside - The cost function is not well defined because randomly at every iteration a bunch of node are turned off
		- So turn off dropout check whether the J is decreasing and then turn on dropout
- Data Augmentation


## **1. L1 and L2 Regularization (Weight Decay)**

### **🔹 L1 Regularization (Lasso)**

- Adds an **L1 penalty** to the loss function:LL1=Loss+λ∑∣w∣L_{L1} = Loss + \lambda \sum |w|LL1​=Loss+λ∑∣w∣
- Encourages **sparsity**, setting some weights to **exactly zero**.
- Useful for **feature selection** and creating compact models.

### **🔹 L2 Regularization (Ridge)**

- Adds an **L2 penalty** (also called **weight decay**) to the loss function:LL2=Loss+λ∑w2L_{L2} = Loss + \lambda \sum w^2LL2​=Loss+λ∑w2
- Helps prevent large weight values, leading to **smooth decision boundaries**.
- Common in CNNs as it distributes weight values evenly.

### **🔹 Elastic Net Regularization**

- Combines both **L1 and L2** penalties:L=Loss+λ1∑∣w∣+λ2∑w2L = Loss + \lambda_1 \sum |w| + \lambda_2 \sum w^2L=Loss+λ1​∑∣w∣+λ2​∑w2
- Provides a balance between **feature selection (L1)** and **weight control (L2)**.

---

## **🔥 2. Dropout Regularization**

- **Randomly drops neurons** during training with probability ppp.
- Forces the network to **learn multiple independent features**, reducing overfitting.
- Applied in **fully connected layers and some convolutional layers**.

**Mathematical Representation:**

ydrop=y1−p (During training)y_{drop} = \frac{y}{1 - p} \text{ (During training)}ydrop​=1−py​ (During training)

- Popular in architectures like **AlexNet, VGG, Inception, and ResNet**.

---

## **🔥 3. Batch Normalization (BN)**

- Normalizes activations across a mini-batch to have **zero mean and unit variance**.
- Helps **stabilize training, improve convergence speed**, and **reduce internal covariate shift**.
- Acts as a form of **regularization by reducing reliance on weight initialization**.
- After fully connected and before activation
![[Pasted image 20250306011721.png]]
- During test a single empirical mean of activation is used estimated during training
- Used in **ResNet, Inception, MobileNet, and EfficientNet**.

---

## **🔥 4. Data Augmentation (Implicit Regularization)**

- **Increases training data** by applying transformations such as:
    - **Rotation, flipping, cropping, scaling, color jittering, and elastic distortions**.
- Prevents overfitting by ensuring the model **generalizes well to unseen variations**.
- Used in **ImageNet-trained models, self-supervised learning, and GANs**.


### **Features of the First Four Regularization Techniques in CNNs**

---

## **1. L1 & L2 Regularization (Weight Decay)**

### **🔹 Features:**

✅ **Prevents overfitting** by adding a penalty to large weights.  
✅ **L1 Regularization (Lasso):** Encourages **sparsity** by making some weights exactly **zero**.  
✅ **L2 Regularization (Ridge):** Ensures **smooth weight distribution**, reducing model complexity.  
✅ **Elastic Net:** Combines L1 & L2 for **balanced feature selection and weight control**.  
✅ Used in **deep CNNs like VGG, ResNet, and Inception** to control complexity.

---

## **2. Dropout Regularization**

### **🔹 Features:**

✅ **Prevents co-adaptation** of neurons by randomly **dropping them** during training.  
✅ **Reduces overfitting** by forcing the network to learn **redundant** and **robust** features.  
✅ **Helps in fully connected layers** (e.g., in AlexNet & VGG).  
✅ Improves generalization by ensuring neurons **don’t become too dependent on others**.  
✅ **Dropout rate (p)** is typically set between **0.2 - 0.5** in practice.

Current Status
- Against 
	- Convergence time high
	- made redundant by BN 
	- not needed for large dataset or augmented dataset
- In Favor
	- RNN , Wide Residual Networks
---
## DropConnect
- Drop some connections
- 
---
## **3. Batch Normalization (BN)**

### **🔹 Features:**

✅ **Normalizes activations** across a mini-batch to **reduce internal covariate shift**.  
✅ **Speeds up convergence** → Reduces dependence on weight initialization.  
✅ **Acts as a regularizer** by adding **slight noise** to activations during training.  
✅ **Improves generalization** → Helps deep networks like **ResNet, MobileNet, and Inception**.  
✅ **Works well with deep networks** (helps stabilize gradient flow in very deep CNNs).

---

## **4. Data Augmentation**

### **🔹 Features:**

✅ **Artificially increases dataset size** using **image transformations** (rotation, flipping, cropping, etc.).  
✅ **Improves model robustness** by exposing it to **varied versions of data**.  
✅ **Works best in vision tasks** (e.g., ImageNet models, self-supervised learning).  
✅ Helps CNNs generalize better to **unseen variations** in real-world data.  
✅ **Common in object detection, segmentation, and classification** tasks.

---

### **🔥 Summary of Features**

|**Regularization Type**|**Key Features**|
|---|---|
|**L1 & L2 Regularization**|Reduces overfitting, encourages sparsity (L1) or smooth weights (L2).|
|**Dropout**|Randomly drops neurons, prevents co-adaptation, enhances generalization.|
|**Batch Normalization**|Normalizes activations, stabilizes training, speeds up convergence.|
|**Data Augmentation**|Artificially increases dataset, improves robustness, prevents overfitting.|

Would you like **code implementations** for any of these? 🚀


## Transfer Learning
![[Pasted image 20250306012109.png]]