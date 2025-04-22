
### AutoEncoders
- Unsupervised Learning algorithm for learning a lower-dimensional feature representation from unlabeled training data

![[Pasted image 20250420201544.png]]

#### Training of Autoencoders
- Train such that features can be used to reconstruct original data
![[Pasted image 20250420213045.png]]
- Throw away the decoder after training


#### Use of Encoders
![[Pasted image 20250420213208.png]]
- Train the supervised learning algorithm based on the features encoded and not the input data


### Generative Adverserial Networks
- Generative - Learn to generate samples
- Discriminator - Learns to distinguish between generated and real samples

![[Pasted image 20250420213628.png]]


#### Jensen Shannon Divergence

![[Pasted image 20250420233223.png]]
- JSD is often used as loss function used in Generative Adverserial Models
- Models that try to generate distribution as closely related to True distribution
- Symmetric Version of KL divergence
- Average of the divergence between two distributions and reverse


![[Pasted image 20250420233609.png]]


### NSGAN (Non Saturating GAN loss)
- Instead of trying to **fool the discriminator** indirectly by making it unsure,  The generator now **directly tries to maximize the discriminator’s belief that fake data is real**.
![[Pasted image 20250420235517.png]]


#### DCGAN (Deep Convolutional Generative Adverserial Model)
- Same as NSGAN but now the generator and discriminator are made of convolutions instead of simple layer



#### Problems in GAN training
- Stability 
	- Parameters can oscillate or diverge , generator loss does not correlate with sample quality
- Mode Collapse
	- Generator ends up modelling only a small subset of the training data


#### Wasserstein GAN
- Uses wasserstein distance instead of JSD
- Architectural Changes 
	- Removes sigmoid from discriminator
	- Use linear output
- Use of weight clipping
	- #### 1. **Capacity Limitation**

- Clipping makes the critic too **simple** or **restricted**.
    
- It can't properly learn the difference between real and fake distributions.
    
- If you clip too aggressively, critic outputs become **very similar** for both real and fake samples ⇒ **small gradients** ⇒ **generator gets no useful learning signal**.
    

---

#### 2. **Vanishing or Exploding Gradients**

- If the clipping range is:
    
    - **Too narrow**: weights shrink ⇒ **vanishing gradients**
        
    - **Too wide**: critic becomes unconstrained ⇒ **unstable gradients**
        
- Either case can result in poor learning, especially for the **generator**, which relies on critic gradients to improve.
    

---

#### 3. **Weight Oscillations**

- Because weights are clipped after each update, they **bounce** between extremes rather than converging smoothly.
    
- This leads to **noisy loss curves** and **instability in both generator and critic** training.
    

---

#### 4. **Discriminator Loss Gets Stuck**

- The **critic loss** may start to **decrease at first**, but then **plateau** or even **increase**, not because the model is learning better, but because the critic is now too weak to distinguish anything.
    
- This can look like:
    
    - Critic loss becomes very small or flat
        
    - Generator loss improves initially, then collapses


### WGAN-GP
- remove weight clipping instead add gradient penalty

### LSGAN
- Regression style loss

|Aspect|**Vanilla GAN**|**Progressive GAN**|
|---|---|---|
|**Training Strategy**|Train **full-resolution** model from the start (e.g. 128×128)|Start with **tiny resolution** (e.g. 4×4), **grow progressively**|
|**Resolution during training**|Fixed throughout training|Gradually increases (4×4 → 8×8 → 16×16 → ... 1024×1024)|
|**Loss Function**|Binary Cross Entropy Loss (original GAN loss)|WGAN-GP loss (preferred) or LSGAN (more stable than BCE)|
|**Downsampling/Upsampling**|Strided convolutions (downsample) and transpose convs (upsample)|Average pooling for downsampling, nearest-neighbor upsampling|
|**Normalization**|Batch Normalization|Pixel-wise normalization (generator) + minibatch stddev (discriminator)|
|**Training Stability**|Very sensitive, easily destabilized|Much **more stable** training, especially for big images|
|**Artifacts**|Checkerboard artifacts (transpose convs)|Cleaner images (no checkerboard from upsampling)|
|**Generator Updates**|Direct updates|Also keeps an **Exponential Moving Average** (EMA) for smooth outputs|
|**Mode Collapse**|Can happen easily|Minibatch stddev reduces it|

| Feature                | Progressive GAN             | StyleGAN                               |
| ---------------------- | --------------------------- | -------------------------------------- |
| Latent input           | Fed directly into generator | Mapped to `w` and used via AdaIN       |
| Feature control        | Hard to isolate             | Can control styles at different layers |
| Image diversity        | Good                        | **Higher**, with noise + AdaIN         |
| Face attribute editing | Difficult                   | Possible with interpolation in W space |
| Disentanglement        | Limited                     | Improved due to mapping + style layers |