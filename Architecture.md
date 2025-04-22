
## LENET
- Digit Classification

- Input - 32 * 32
- Conv - Filter 1 - 6 of 5x5
- Pooling - Avg - 2X2 and stride 2
- Conv -Filter 1 - 16 of 5x5
- Avg Pooling - 2x2 and stride 2
- Conv Layer - 120 x 1 x 1
- 84 Neurons
- Softmax Layer
- Activation function in every conv layer and neural network is tanh



## ALEXNET

- 7 Layers
- Local Response Normalisation
- ReLU , Dropouts
- Data Augmentation
- 60 million parameters
![[Pasted image 20250305223634.png]]
- Trained on GTX 580 had only 3GB of memory . So network spread across 2 GPUs 
- Input = 227 x 227 x 3
- Multiple pathways increased the computational efficiency by splitting the network into two pathways

### **Key Features of AlexNet:**

1. **Deep Architecture:**
    
    - Consists of **8 layers** (5 convolutional layers + 3 fully connected layers).
2. **ReLU Activation:**
    
    - Uses **Rectified Linear Unit (ReLU)** instead of sigmoid or tanh, which speeds up training and prevents vanishing gradients.
3. **Overlapping Max Pooling:**
    
    - Uses max pooling with an overlap to reduce dimensionality while retaining important features.
4. **Dropout for Regularization:**
    
    - Introduced **dropout** in fully connected layers to prevent overfitting.
5. **Data Augmentation:**
    
    - Uses techniques like **random cropping, flipping, and color variation** to increase dataset size and improve generalization.
6. **Large Kernel Sizes:**
    
    - Uses **11×11, 5×5, and 3×3 kernels** in convolutional layers to capture hierarchical features.
7. **Two GPU Training:**
    
    - Designed to be trained on **two GPUs** in parallel for faster computation.
8. **Softmax Classifier:**
    
    - Uses **Softmax** in the final layer for classification.

## ZFNet
- CONV 1 = 11 x 11 stride 4 to 7 x 7 stride 2
- CONV 3,4,5 = 384,384,256 to 512,1024,512


## VGGNet
- Deeper network
- Small filters 
![[Pasted image 20250305225004.png]]

### **Why Use Small Filters and Deeper Networks in CNNs?**

In Convolutional Neural Networks (CNNs), using **smaller filters** (e.g., **3×3** instead of **5×5** or **7×7**) and **deeper architectures** (more layers) has several advantages in terms of **computational efficiency, feature extraction, and generalization**.

---

## **1. Advantages of Small Filters (e.g., 3×3) Over Large Filters (e.g., 5×5, 7×7)**

### **a) Capturing More Local Features Efficiently**

- **Smaller filters capture fine-grained spatial details.**
- **Stacking multiple 3×3 layers** increases the **receptive field** while keeping computations efficient.
- Instead of a **5×5** filter, using **two 3×3 filters** covers the same area but introduces **non-linearity** and depth.

📌 _Example:_

- **One 5×5 filter** → Covers a **5×5** area.
- **Two stacked 3×3 filters** → Covers a **5×5** area with **two layers of non-linearity** (ReLU), improving feature extraction.

### **b) Fewer Parameters → Less Overfitting**

- A **5×5 filter** has **25 parameters** while a **3×3 filter** has only **9 parameters**.
- Fewer parameters **reduce memory usage** and prevent **overfitting**, especially for small datasets.

📌 _Example:_

- **5×5 Conv Layer:** 25 parameters per filter.
- **3×3 Conv Layer:** 9 parameters per filter → **More efficient and generalizable.**

### **c) Computational Efficiency (Less FLOPs, Faster Training)**

- Smaller filters require **fewer multiplications and additions**.
- **3×3 filters reduce the number of floating-point operations (FLOPs)** compared to 5×5 or 7×7 filters.

📌 _Example:_

- A **5×5 filter** requires **25 multiplications per pixel**, whereas a **3×3 filter** requires only **9**.
- Two **stacked 3×3 filters** (total **18** multiplications) still require fewer computations than a **single 5×5 filter (25 multiplications)**.

---

## **2. Advantages of Deeper Networks Over Wider Networks**

### **a) Increased Hierarchical Feature Extraction**

- **Shallow networks with large filters** extract low-level features only.
- **Deeper networks with smaller filters** extract **both low-level (edges, textures) and high-level (shapes, objects) features** progressively.

📌 _Example:_

- First layers learn **edges and corners**.
- Middle layers learn **shapes and patterns**.
- Deeper layers learn **object-level representations** (e.g., faces, cars).

### **b) Improved Non-Linearity and Expressive Power**

- More layers mean **more activation functions (ReLU, LeakyReLU, etc.)** → Increased **model capacity**.
- **Deeper networks can approximate complex functions** better than shallow networks.

📌 _Example:_

- A **shallow CNN** struggles to differentiate handwritten digits with **similar strokes** (e.g., 3 vs. 8).
- A **deeper CNN** learns detailed patterns (e.g., stroke orientation, loops) to distinguish digits.

### **c) Easier Optimization via Techniques like Batch Normalization & Residual Connections**

- Deep networks used to suffer from **vanishing gradients**.
- **Batch Normalization (BN)** and **ResNet (Residual Connections)** solve this by stabilizing gradients.

📌 _Example:_

- **ResNet-50 (50 layers)** outperforms a **simple 10-layer CNN** because **skip connections prevent gradient decay**.

---

## **Real-World Implementations Using Small Filters and Deep Networks**

### **1. VGG-16 (Small Filters + Deep Architecture)**

- Uses only **3×3 filters**, stacked deeply.
- Outperforms older models using **larger filters (e.g., 5×5, 7×7)**.

### **2. ResNet (Deep + Skip Connections)**

- Uses multiple **3×3 layers** with **residual connections** to go as deep as **152 layers** without vanishing gradients.

### **3. Inception (Hybrid Approach)**

- Uses **1×1, 3×3, and 5×5 filters** but mostly relies on **3×3 convolutions** for efficiency.

---

## **Conclusion**

Using **small filters (3×3) and deeper networks** improves **feature extraction, efficiency, and generalization** while reducing computational costs. **Modern CNN architectures (VGG, ResNet, Inception, etc.) rely on this strategy** to achieve state-of-the-art results in computer vision tasks. 




## Network in Network
### **Key Features of Network in Network (NiN):**

1. **MLPConvs (Micro-Net within a CNN)**
    
    - Instead of a single convolutional layer, NiN introduces **multiple fully connected (FC) layers (MLPs) within a convolutional layer**.
    - Each convolutional layer consists of a **stack of 1×1 convolutions**, acting as an **MLP at each spatial location**.
2. **1×1 Convolution for Feature Abstraction**
    
    - NiN replaces traditional **fully connected layers** with **1×1 convolutions**, reducing parameters and enhancing non-linearity.
    - Helps **capture complex representations** while maintaining **spatial structure**.
3. **ReLU Activation for Faster Training**
    
    - Uses **Rectified Linear Unit (ReLU)** activation after each 1×1 convolution to accelerate convergence.
4. **Global Average Pooling (GAP) Instead of Fully Connected Layers**
    
    - NiN replaces fully connected layers with **Global Average Pooling (GAP)** to reduce overfitting.
    - Instead of using FC layers for classification, GAP averages each feature map, outputting class probabilities directly.
5. **Fewer Parameters, Less Overfitting**
    
    - Compared to architectures like AlexNet, NiN significantly reduces the number of parameters, preventing overfitting.
6. **Increased Non-linearity and Expressiveness**
    
    - The **stacked MLPs** inside convolutional layers increase the model's learning capacity without adding excessive parameters.


## **Key Features of GoogleNet (Inception v1):**

### **1. Inception Module - Multi-scale Feature Extraction**

- Instead of using a **single filter size** for convolution (like AlexNet), GoogleNet applies **1×1, 3×3, and 5×5 convolutions** in parallel, allowing the network to capture features at different scales.
- Also includes **1×1 convolutions for dimensionality reduction**, which decreases computational cost.

### **2. 1×1 Convolutions for Dimensionality Reduction**

- Before applying **3×3 or 5×5 convolutions**, **1×1 convolutions** reduce the number of channels, reducing computational complexity and preventing overfitting.

### **3. Deep Network with Fewer Parameters**

- Despite having **22 layers**, GoogleNet has **12 times fewer parameters (~5 million) than AlexNet (~60 million)**.

### **4. Global Average Pooling (GAP) Instead of Fully Connected Layers**

- Instead of fully connected layers, GoogleNet uses **Global Average Pooling (GAP)**, which **averages feature maps** before classification.
- This reduces overfitting and improves generalization.

### **5. Auxiliary Classifiers for Better Gradient Flow**

- Two **auxiliary classifiers** are added in intermediate layers to help **gradient propagation** in deep networks.
- Each classifier consists of a **small CNN + Softmax**, helping improve convergence.

### **6. Efficient Use of Computational Resources**

- By using **factorized convolutions (1×1, 3×3, 5×5)** and **parallel paths**, GoogleNet achieves **high accuracy with lower computational cost** compared to previous architectures.



### **Inception V2,V3**
- **Factorized Convolutions**
    
    - Instead of a **large 5×5 convolution**, it uses **two stacked 3×3 convolutions**, reducing computational cost.
    - Instead of a **3×3 convolution**, it uses two **1×3 and 3×1 convolutions** for efficiency.
- **Batch Normalization (BN)**
    
    - BN is added **after every convolutional layer**, stabilizing training and improving convergence speed.

## **Key Features of ResNet:**

### **1. Residual Connections (Skip Connections) – Solves Vanishing Gradient Problem**

- Instead of learning a direct mapping, ResNet learns a **residual function:**F(x)=H(x)−x⇒H(x)=F(x)+xF(x) = H(x) - x \quad \Rightarrow \quad H(x) = F(x) + xF(x)=H(x)−x⇒H(x)=F(x)+x
- **Shortcut connections (skip connections)** add input (`x`) directly to the output of a layer (`F(x)`).
- Helps **preserve gradient flow**, allowing training of very deep networks.

---

### **2. Identity Mapping – Efficient Gradient Propagation**

- Directly passes **input** to later layers, avoiding degradation in very deep networks.
- Makes optimization easier, enabling training of **50, 101, and even 152-layer models**.

---

### **3. Bottleneck Layers – Reducing Computation (ResNet-50 and deeper)**

- Uses **1×1 convolutions** before and after **3×3 convolutions** to **reduce the number of parameters**.
- A typical **bottleneck block**:
    - **1×1 conv (reduce channels) → 3×3 conv → 1×1 conv (restore channels) → Skip connection**

---

### **4. Deep Networks Without Overfitting**
- Deeper Models are harder to Optimise
- ResNet-152 achieved **better performance** than shallow networks like VGG-19 with **fewer parameters**.
- Example comparisons:
    - **VGG-19**: 144 million parameters
    - **ResNet-50**: 25.5 million parameters

![[Pasted image 20250305235144.png]]

![[Pasted image 20250305235432.png]]
![[Pasted image 20250305235532.png]]

## Pre Activated ResNET
### **1. Pre-Activation Order (BN → ReLU → Conv) – Better Gradient Flow**

- Unlike **ResNet-v1**, where the order is **Conv → BN → ReLU**, ResNet-v2 changes it to:
    
    `BN → ReLU → Conv`
    
- This improves **gradient propagation** and makes training deep networks easier.
- Helps **preserve feature information better** in deep architectures.

---

### **2. Improved Residual Block Design**

- Each **bottleneck block** follows the pre-activation order:
    
    `Input → BN → ReLU → 1×1 Conv → BN → ReLU → 3×3 Conv → BN → ReLU → 1×1 Conv → Skip Connection → Output`
    
- Ensures that activations are normalized before every convolution.

---

### **3. Better Training Stability for Very Deep Networks**

- Reduces **internal covariate shift**, making training of **extremely deep networks (e.g., ResNet-200, ResNet-1000) more stable**.
- Helps in better **convergence** compared to ResNet-v1.

---

### **4. No Activation Function After the Addition (Improved Shortcut Connection)**

- In original ResNet, after adding the shortcut connection (`x + F(x)`), ReLU was applied.
- In Pre-Activated ResNet, **no activation function is applied after addition**, allowing better gradient flow.

---

### **5. Outperforms Original ResNet (Especially on ImageNet & CIFAR-10/100)**

- **Achieves better accuracy** on deeper architectures compared to ResNet-v1.
- Helps prevent **performance degradation** seen in very deep models.





## Squeeze and Excitation Net

- To recognise the more informative channel of the feature map if the filter rather than applied across all channels it is applied on single corresponding channel
- To know the importance of one channel we squeeze the feature map in 1x1xC and simplest way to do so is global average pooling
- But doing so does not tell which is more important there we introduce excitation  
- Now we simply scale our coefficient to the corresponding feature map
![[Pasted image 20250306000937.png]]
- After this convolution kernel will pay more attention to the more important channels
- Excitation Operation 
	- FC (W1 * z)
	- Activation Relu 
	- FC (W2 * output)
	- Sigmoid

![[Pasted image 20250306001329.png]]


## Deep Network
- Randomly drop a subset of layers
- bypass with identity
- Use full network at test time

## Wide ResNet
- Reduce number of residual blocks , but increase the number of feature maps

## DenseNet
- Connect them all

## ResNext
- Increase width of residual block through multiple parallel pathways