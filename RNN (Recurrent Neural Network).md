
### Why?
- Not all data is independent 
![[Pasted image 20250421072548.png]]


#### Recurrent Unit
![[Pasted image 20250421072648.png]]

#### Vanilla RNN
![[Pasted image 20250421072803.png]]



### LSTM
![[Pasted image 20250421112215.png]]
- To solve the problem of Vanishing/Exploding Gradient Descent LSTM was introduced
- Uses Sigmoid and tanh activation Functions



![[Pasted image 20250421114321.png]]

- Green line that runs all the way accross the top of the unit is callled CELL STATE and represents long term memory
- Although long memory can modified using multiplication and addition you will notice there are no weighs and biases 
- The lack of weights and baises allows the Long term memories to flow through a series of unrolled units without causing the gradient to explode or vanish


- Pink line represents short term memories or hidden states and are directly connected to weights that modify them

- First stage of LSTM determines what percentage of the Long Term Memory is remembered . Terminology Alert - Forget gate even though it determines the remember percentage

- Second Stage 2 blocks
	- One determines Potential long term memory from the current input
	- Second determines what percentage of this memory should be remember
- Terminology Alert - Input Gate

- The remembering blocks have sigmoid activation while the potential long term memory block has tanh activation


- Stage 3 Short term memory updates
	- Potential short term memory
	- Percentage of short term memory to remember

- New short term memory is the output of LSTM and this called output gate 


In summary the LSTM uses two separate paths for long term memory and short term memory which theoritically allows us to avoid the vanishing/exploding gradient problem and unroll the recurrent network more times to accommodate longer sequences of input data.


---

### 🍌 What is GNMT? 

GNMT = **Google's big brain** for translating one language to another (like English ↔ Hindi).  
It's like a **super-powered LSTM-based machine translator** 🧠💬.

---

### 🧱 Step-by-step like LEGO blocks:

#### 1. **LSTM Encoder-Decoder**

You already know this part, monkey:

- 🛠 **Encoder LSTM**: Reads the full input sentence (like "I love bananas") and squishes it into a big thought vector 🧠💭.
    
- 🎯 **Decoder LSTM**: Uses that thought vector to write out the translated sentence (like “Main kele pasand karta hoon”).
    

GNMT **still uses this idea**, but makes it **bigger, deeper, and smarter**.

---

### 🧠 So what’s new in GNMT?

#### 🍰 2. **Stacked LSTM Layers**

> Instead of 1 LSTM layer... GNMT stacks **8 LSTM layers!**

Like stacking 8 monkey friends on each other's shoulders — you can see higher 🍌🌳

- First layer learns basic stuff (words)
    
- Higher layers understand grammar, context, emotion, etc.
    

---

#### 🔁 3. **Bi-directional Encoder**

Remember **Bi-RNN**? GNMT uses that in the encoder.

So it reads the sentence:

- Forward → from start to end
    
- Backward ← from end to start
    

💡 This way it **understands the whole sentence better**.

---

#### 👀 4. **Attention Mechanism**

Your LSTM memory gets tired with long sentences, right?

GNMT adds **attention**:

> The decoder **looks back** at different parts of the input sentence when translating each word.

📌 It’s like a monkey reading a sentence but glancing back when confused.

So instead of relying only on one “thought vector,” the decoder can look at:

- “Where in the input should I focus for this word?”
    

---

#### 📊 5. **Residual Connections & Normalization**

These are like monkey shortcuts 🐒 → 🐵 — helping training go faster and reducing “forgetfulness.”

---

### 🧪 Summary

|GNMT Feature|What It Does|
|---|---|
|Bi-directional Encoder|Understands full context|
|8-layer stacked LSTM|Deep understanding from word to grammar|
|Attention mechanism|Focuses on the right parts during translation|
|Residual connections|Easier training, fewer vanishing gradients|
|Decoder with Attention|Smarter, context-aware word generation|

---

### 💬 In Monkey Words:

> GNMT is like a **super LSTM monkey** 🦍 with 8 brains, two eyes looking both ways, and magical focus glasses 🕶 to zoom in on important words while translating.

---

Wanna see how GNMT evolved into Transformers next? Or how attention works with a cute example?


https://chatgpt.com/share/6805f95b-b61c-8007-8856-90ecaa4d7751