
### Tranformer Architecture

![[Pasted image 20250422064250.png]]

#### Encoder Stack and Decoder Stack
- Input Embeddings
- then Positional Embeddings
- Key (Wk . X ) , Value (Wv. X) , Query (Wq . X)
- then Attention weights = softmax(Q.K / sqrt(dk))
- Attention = AV => Y
- ![[Pasted image 20250422065315.png]]
- Multihead attention alllows the model to jointly attend to information from different representation subspaces at different positions.

##### Application of Attention
1. Self Attention in Encoder --- Q , K , V are from the same sentence i.e output of previous layer in the encoder. Each position in encoder can attend to all positions in the previous layer of the encoder
2. Encoder - Decoder Attention in Decoder --- Q are from previous layer of Decoder whereas the K, V are from the encoder. This allows every position in the decoder to attend to every position in the input sentence
3. Self(Masked) Attention in Decoder --- this allows each position in decoder to attend to all position in the decoder up to and including current position


### Image Vision Transformer
![[Pasted image 20250422062609.png]]

- Image input is of size H x W x C
- The standard transformer receives 1D token embeddings as an input 
- So  here the 2D images is converted into flattened 2D patches of Size P x P
- N = H x W / P x P 
- The transformer uses a Latent Vector of size D so these patches are flattened and mapped to D dimensions with a trainable linear projection . Called as Patch Embeddings
- Then these patch embeddings are combined with positional embeddings
	-  A **cat's face** is not just a collection of ears, eyes, and nose — their **relative positions** matter!
	- Without positional encoding, a cat’s nose could be next to a tail and the model wouldn’t care
- Now these embedded patches are passed through transformer encoder
	- First Normalisation
	- Then Multihead Attention
	- Then sum of this output and residual link before the norm
	- Then Norm
	- Then MLP 
	- Add with residual link before second norm
- MLP contains 2 layers with GELU non linearity



### BERT (Bidirectional Encoder Representation for Tranformers)
Alright! Let's break down **BERT** — like you're hearing it for the first time but already know some NLP basics like word embeddings and transformers.

---

### 🧠 What is BERT?

**BERT** stands for:

> **B**idirectional **E**ncoder **R**epresentations from **T**ransformers

It’s a **pretrained language model** that understands the **context of words in a sentence by looking at both left and right** sides — unlike older models that read only left-to-right or right-to-left.

---

### 📦 Architecture

- BERT is **only the encoder** part of the Transformer.
    
- It stacks multiple encoder layers (12 for BERT-Base, 24 for BERT-Large).
    
- Each encoder has:
    
    - Multi-head self-attention
        
    - Feed-forward layers
        
    - Layer norm, residual connections
        

---

### 🔁 Bidirectional: Why It’s Special

Traditional models like GPT (left-to-right) or ELMo (two unidirectional models) look only at past (or future) words.

**BERT reads entire sequences at once** — so it can learn truly contextual meaning:

> “He went to the **bank** to deposit money.”  
> vs.  
> “He sat on the **bank** of the river.”

BERT understands **which “bank”** you mean because it sees **both left and right context**.

---

### 🧪 Pretraining Tasks

BERT is trained on massive text (like Wikipedia) using two clever self-supervised tasks:

1. **Masked Language Modeling (MLM)**
    
    - Randomly masks 15% of the tokens.
        
    - The model tries to predict the original words from the surrounding context.
        
    - Example:  
        Input: “The cat sat on the **[MASK]**.”  
        Predict: “mat”
        
2. **Next Sentence Prediction (NSP)**
    
    - Given sentence A and sentence B, predict if B logically follows A.
        
    - Helps BERT understand sentence relationships — useful for QA, inference, etc.
        

---

### 🛠 Fine-Tuning

Once pretrained, BERT can be **fine-tuned** on downstream tasks:

- Classification (sentiment, topic)
    
- Named Entity Recognition (NER)
    
- Question Answering (SQuAD)
    
- Text similarity
    
- Anything NLP!
    

You just add a small output layer, train it on a specific dataset for a few epochs, and done.

---

### 💪 Why BERT was a Game Changer

- Set **state-of-the-art** results on 11 NLP benchmarks.
    
- Made **pretraining + fine-tuning** the new standard in NLP.
    
- Enabled transfer learning for language — just like ImageNet did for vision.
    

---

### 🧠 TL;DR

| Feature           | BERT                   |
| ----------------- | ---------------------- |
| Directionality    | Bidirectional ✅        |
| Transformer Part  | Encoder only ✅         |
| Pretraining Tasks | MLM + NSP ✅            |
| Use Cases         | Almost all NLP tasks ✅ |

---

Let me know if you want visuals, comparisons with GPT, or how BERT’s successors (like RoBERTa, DistilBERT, etc.) improved on it!




#### Transformer vs Decoder Only 
1. Transformer uses one unit to encode and another to decode whereas Decoder Only Transformer has a single unit to decode and encode
2. Transformer uses 2 types of attention Self Attention and Encoder Decoder Attention during inference but decoder only transformer uses single type of attention Masked Self Attention
3. During Training Transformer uses a masked self Attention but only on the output and decoder uses it all the time