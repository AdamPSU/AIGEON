## Skip-gram Algorithm  

At a high-level, the algorithm creates word representations that accurately "predict" the surrounding context. 

### How? 

1. initialize vectors $w_1, w_2, ..., w_n$, each of size $d$, with values $\approx 0$. 
2. Maximize the logged probability that some "center" word $w_t$ predicts some "context" word $w_{t+i}$ $\to p(w_{t+i}|w_{t})$
3. Gradient descent over many epochs to update the learned embeddings 

By iterating over millions of words scraped from newspapers, the authors found embeddings that captured the semantics of natural language, as shown below:  

> $vec(\text{"Germany"}) - vec(\text{"Berlin"}) + vec(\text{"France"}) = vec(\text{"Paris"})$

The authors propose maximizing the loss function

$$\frac{1}{N} \sum_{t=1}^{N}\sum_{i=-C}^{C}log(p(w_{t+i}|w_t)), \quad i \neq 0$$

which raises a natural question: how is this probability defined? We choose to softmax the dot product between the context ($W_O$) and center vectors ($W_I$). 

$$p(w_O|w_I) = \text{softmax}(W_{O}^\top \cdot W_{I})$$

However, because the softmax computation is proportional to the vocabulary size $V$, this algorithm is pretty inefficient. 

## Extensions 
Further extensions to this work include a repurposed NCE loss funcion called NEG, and efficient subsampling techniques, which aim to balance the frequencies between common and uncommon words (i.e. "the" vs. "Paris"). 




