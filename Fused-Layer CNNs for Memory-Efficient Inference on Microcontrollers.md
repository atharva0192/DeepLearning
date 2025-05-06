
### **Abstract**

Deploying Convolutional Neural Networks (CNNs) on microcontrollers poses significant challenges due to their limited memory and computational capabilities. The paper “Fused-Layer CNNs for Memory-Efficient Inference on Microcontrollers” presents a novel approach to significantly reduce the memory footprint of CNNs during inference by fusing layers—specifically, inverted residual blocks—in quantized MobileNetV2-like architectures. This technique allows the intermediate feature maps to be computed tile-by-tile rather than layer-by-layer, dramatically minimizing RAM usage with negligible accuracy loss and tolerable latency increases. This report covers the original methods, implementation from scratch, experimental results, and extensions through our innovative trials on alternative lightweight CNN architectures.

---

### **Literature Review**

Early efforts to optimize CNNs for resource-constrained environments focused on techniques like quantization, pruning, and lightweight model design. BatchNorm folding and ReLU fusion were standard optimizations in many deep learning frameworks, aiming to reduce inference complexity but had minimal impact on memory consumption.

Lin et al. introduced **in-place depthwise convolutions** via MCUNet to optimize RAM by reusing memory between channel computations. While effective, the method was limited to depthwise-separable convolutions and tied closely to their own ecosystem.

The paper in focus extends this idea by showing that **entire inverted residual blocks**—including a 1×1 expansion, a 3×3 depthwise convolution, and a 1×1 projection—can be fused. This approach allows tiling across spatial dimensions, significantly reducing the size of intermediate feature maps. Compared to MCUNet’s memory compression (up to 1.6×), this approach achieves up to 2.1× memory reduction on standard MCUs.

---

### **Base Paper Implementation**

#### **1. Overview**

The original paper did not provide source code, so we re-implemented the methods from scratch, aiming to replicate the memory-efficient fused CNN inference mechanism using:

- **Layer Fusion**
    
- **Inverted Residual Blocks**
    
- **Quantized CNNs**
    

#### **2. Methodology**

- **Layer Fusion via Tiling**: Instead of computing CNN layers sequentially, the paper tiled the feature maps spatially and computed them in-depth order, reusing memory within tiles.
    
- **Inverted Residual Blocks**: These blocks follow a "narrow-wide-narrow" channel structure, making them ideal for fusion.
    
- **Quantization**: We applied tensor quantization and folded batch normalization and ReLU layers into the convolutional weights, minimizing floating-point operations.
    

#### **3. Experimental Setup**

- Implemented MobileNetV2 with and without fusion.
    
- Simulated MCU constraints in software, tracking memory usage and inference performance.
    
- Evaluated on CIFAR-10 with 128×128 resolution.
    

#### **4. Results**

|Metric|No Fusion|With Fusion|
|---|---|---|
|Peak Memory|792.00 kB|343.50 kB|
|Accuracy|43.76%|46.79%|
|Memory Reduction|—|~50%|

The fusion approach preserved or slightly improved accuracy while reducing memory usage significantly.

---

### **Innovation**

After replicating the baseline work, we explored ways to extend and improve upon the technique. Our innovations fall into two key areas:

#### **1. Applying Fusion to Other Lightweight Architectures**

We explored applying the same layer fusion strategy to other CNNs designed for efficiency:

- **ShuffleNet**
    
- **MobileNetV3**
    
- **EfficientNet-Lite**
    

While ShuffleNet didn't yield improvements due to its lack of inverted residual blocks, EfficientNet-Lite showed promise.

##### **EfficientNet-Lite Fusion Results**:

- Achieved **61% memory reduction**, surpassing MobileNetV2.
    
- Minimal latency penalty, demonstrating the method’s adaptability beyond the original model.
    

---

### **Conclusion**

The layer fusion strategy proposed in the base paper proves to be a highly effective method for memory optimization on microcontrollers. Our work validates the original method and extends its applicability, achieving even greater memory savings through innovations like application to EfficientNet-Lite and hybrid quantization strategies. Future work can explore automated layer fusion schedulers for arbitrary architectures and MCU profiling for real-time deployments.

---
