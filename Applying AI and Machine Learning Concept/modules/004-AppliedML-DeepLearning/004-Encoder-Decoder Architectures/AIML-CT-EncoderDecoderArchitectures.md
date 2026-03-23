# Encoder-Decoder Architectures

**Estimated Time:** 10 Minutes

---

## Introduction

Up to this point, the sequence models you have studied take a sequence in and produce either a single output (classification) or an output at each time step (tagging). But some of the most compelling problems in deep learning require transforming one variable-length sequence into a *different* variable-length sequence: translating English to French, summarizing a document, converting speech to text, or generating a reply in a chatbot. The input and output sequences can be completely different lengths, and there is no simple one-to-one mapping between their elements.

The **Encoder-Decoder** architecture (also called **sequence-to-sequence**, or **seq2seq**) was designed precisely for this class of problems. In this reading you will learn how the encoder and decoder work together, how information flows between them, and the key design decisions involved.

---

## Core Concepts

### The Big Picture

An encoder-decoder model has two distinct components:

1. **Encoder** -- reads the entire input sequence and compresses it into a fixed-size representation called the **context vector** (sometimes called the "thought vector"). This context vector is supposed to capture the meaning of the input sequence.
2. **Decoder** -- takes the context vector and generates the output sequence one token at a time, using its own hidden state plus the context from the encoder.

The two components are trained jointly, end to end, so the encoder learns to produce representations that are useful for the decoder's task.

### The Encoder

The encoder is typically a recurrent network (usually an LSTM or GRU) that processes the input sequence step by step. Its final hidden state becomes the context vector.

In pseudocode:

```
function Encoder(source_tokens):
    embedded = Embedding(source_tokens)
    _, (final_hidden, final_cell) = LSTM(embedded)
    return final_hidden, final_cell   # the context vector
```

The encoder does not produce any predictions. Its only job is to read and compress.

### The Decoder

The decoder is another recurrent network that generates the output sequence one token at a time. At each step, it receives:

- The previously generated token (or, during training, the ground-truth previous token -- a technique called **teacher forcing**).
- The hidden state from the previous decoding step.
- The context vector from the encoder (used to initialize the decoder's hidden state).

In pseudocode:

```
function Decoder(token, hidden, cell):
    embedded = Embedding(token)
    output, (hidden, cell) = LSTM(embedded, (hidden, cell))
    prediction = Linear(output)         # scores over target vocabulary
    return prediction, hidden, cell
```

Notice that the decoder produces one prediction per call. During inference, you feed each predicted token back as the input for the next step, repeating until the model generates a special end-of-sequence token.

### Wiring Them Together

The seq2seq model connects the encoder and decoder by passing the encoder's final hidden state as the decoder's initial hidden state:

```
function Seq2Seq(source, target, teacher_forcing_ratio=0.5):
    hidden, cell = Encoder(source)        # Encode the input

    outputs = []
    current_token = START_TOKEN

    for t = 1 to target_length:
        prediction, hidden, cell = Decoder(current_token, hidden, cell)
        outputs.append(prediction)

        if random() < teacher_forcing_ratio:
            current_token = target[t]       # teacher forcing: use ground truth
        else:
            current_token = argmax(prediction)  # use model's own prediction

    return outputs
```

### Teacher Forcing

During training, you have a choice at each decoder step: feed in the ground-truth previous token (teacher forcing) or feed in whatever the model predicted. Teacher forcing speeds up training dramatically because the decoder does not have to recover from its own mistakes. However, it creates a mismatch between training (where the decoder always sees correct inputs) and inference (where it must rely on its own, potentially wrong, predictions).

A common compromise is to use teacher forcing with some probability (e.g., 50%) and let the model use its own predictions the rest of the time. This is called **scheduled sampling**, and it helps the decoder learn to be robust to its own errors.

### The Bottleneck Problem

The original seq2seq architecture has a significant limitation: the entire input sequence must be compressed into a single fixed-size context vector. For short sentences this works reasonably well, but for longer inputs the context vector simply cannot hold all the necessary information. Performance degrades as input length increases.

This bottleneck is what motivated the development of **attention mechanisms**, which allow the decoder to look back at all of the encoder's hidden states (not just the final one) at each decoding step. Attention is the subject of the next module, and it eventually led to the Transformer architecture that dominates modern NLP.

### When to Use Encoder-Decoder Models

Encoder-decoder architectures are the right tool when:

- The input and output are both sequences, potentially of different lengths.
- There is no simple alignment between input and output positions.
- Examples include machine translation, text summarization, conversational response generation, and speech-to-text.

If input and output lengths match and there is a natural one-to-one correspondence (e.g., part-of-speech tagging), a simpler sequence labeling model is usually more appropriate.

---

## Connecting to Practice

In the upcoming exercise, you will implement a complete seq2seq model. You will build the encoder and decoder as separate components, wire them together, and train the combined model on a small parallel corpus. You will experiment with teacher forcing ratios and observe how they affect training stability and output quality. This hands-on experience will prepare you for the attention mechanisms and Transformer architectures covered in the next module.

---

## Further Learning & Resources

### Documentation

1. [Sutskever et al., 2014: Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) -- The original paper that introduced the LSTM-based encoder-decoder model for machine translation.
2. [Dive into Deep Learning: Encoder-Decoder Architecture](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html) -- Textbook chapter with mathematical derivations and implementation details for the seq2seq framework.
3. [Deep Learning Book, Chapter 10: Sequence Modeling](https://www.deeplearningbook.org/contents/rnn.html) -- Comprehensive coverage of recurrent architectures including seq2seq models.

### Interactive Resources

1. [Dive into Deep Learning: Seq2Seq (Interactive)](https://d2l.ai/chapter_recurrent-modern/seq2seq.html) -- Interactive Jupyter-based chapter with runnable code implementing seq2seq translation from scratch.
2. [Jay Alammar: Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) -- Illustrated, interactive walkthrough of how encoder-decoder models process and generate sequences step by step.
