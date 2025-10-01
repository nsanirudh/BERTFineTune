# Fine-Tuning BERT for Quora Insincere Questions

This project fine-tunes the TensorFlow Hub **BERT Base, Uncased (L-12 H-768 A-12)** model to detect insincere questions from the Quora dataset. The workflow mirrors the Coursera guided project while updating it for the current toolchain used in this repository.

## Environment Setup (Python 3.11.13)

1. Create and activate the environment:
   ```bash
   conda create -n bert python=3.11.13
   conda activate bert
   ```
2. Export the legacy Keras flag **before importing TensorFlow** to keep TF-Hub `KerasLayer` compatible with Keras 3:
   ```bash
   export TF_USE_LEGACY_KERAS=1
   ```
3. Install dependencies (matching the notebook kernel):
   ```bash
   pip install --upgrade pip
   pip install \
       tensorflow==2.20.0 \
       tensorflow-hub==0.16.1 \
       tensorflow-text==2.20.0 \
       tensorflow-addons==0.23.0 \
       pandas scikit-learn matplotlib
   ```
   > Tip: the repository already vendors the `tensorflow/models` source tree under `models/`. If it ever goes missing, re-clone it with `git clone --depth 1 -b v2.3.0 https://github.com/tensorflow/models.git`.
4. Optional GPU fallback: if TensorFlow cannot find `libdevice.10.bc` or `ptxas`, either install a matching CUDA toolkit or run the notebook on CPU by exporting `TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false` (and optionally hiding GPUs with `tf.config.set_visible_devices([], "GPU")`).
5. Launch Jupyter or Colab and open `BERT Fine Tune.ipynb`.

Key versions recorded from the active kernel:

| Component      | Version |
|----------------|---------|
| Python         | 3.11.13 |
| TensorFlow     | 2.20.0  |
| TensorFlow Hub | 0.16.1  |

## Data Preparation

- Source: [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification/data) mirrored at `https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip`.
- Load the compressed CSV directly with pandas (`low_memory=False`).
- Create a stratified sample for faster experimentation:
  - Training split: 0.75% of the data.
  - Validation split: 0.075% of the remaining examples.
- Convert the text columns to TensorFlow datasets with `tf.data.Dataset.from_tensor_slices` prior to feature mapping.

## Feature Engineering Pipeline

1. **Tokenizer and BERT layer** – Use `hub.KerasLayer` to load the pretrained encoder and expose the accompanying vocab file to the `FullTokenizer` from the Model Garden.
2. **`to_feature` helper** – Wraps the Model Garden `classifier_data_lib.InputExample` utilities to convert raw text into `(input_ids, input_mask, segment_ids, label)` tuples given `max_seq_len=128`.
3. **`to_feature_map` wrapper** – Adapts `to_feature` for `tf.data` by calling it through `tf.py_function`, fixes tensor shapes, and returns a feature dictionary keyed by `input_word_ids`, `input_mask`, and `input_type_ids` plus the label scalar.
4. **Dataset pipeline** – Map the conversion function, shuffle with a buffer of 1,000 examples, batch and prefetch:
   ```python
   train_data = (train_data
       .map(to_feature_map, num_parallel_calls=tf.data.AUTOTUNE)
       .shuffle(1000)
       .batch(32, drop_remainder=True)
       .prefetch(tf.data.AUTOTUNE))

   val_data = (val_data
       .map(to_feature_map, num_parallel_calls=tf.data.AUTOTUNE)
       .batch(32, drop_remainder=True)
       .prefetch(tf.data.AUTOTUNE))
   ```

## Understanding BERT and Input Encodings

BERT (Bidirectional Encoder Representations from Transformers) stacks 12 transformer encoder blocks, each pairing multi-head self-attention (12 heads) with feed-forward networks sized to a 768-dimensional hidden state. Pre-training uses masked-language modelling and next-sentence prediction so the encoder learns bidirectional context before any downstream fine-tuning. The TensorFlow Hub module `bert_en_uncased_L-12_H-768_A-12/2` exposes the frozen encoder weights and pooling layer.

"Uncased" indicates text is lowercased and stripped of accent markers before tokenization; the vocabulary merges case variants (e.g. “Apple” and “apple”) into a single token id, which matches the preprocessing inside the Hub module.

### Input features produced for BERT
- `input_word_ids`: WordPiece token ids referencing the 30K-item vocabulary shipped with the Hub module. Each sentence is tokenized into subword units, wrapped with `[CLS]` and `[SEP]`, and padded with zeros up to `max_seq_len`.
- `input_mask`: A binary attention mask (1 for real tokens, 0 for padding) that blocks the model from attending to padded positions during self-attention.
- `input_type_ids`: Also called segment ids. BERT Base was trained on sentence pairs; values are 0 for tokens from the first segment and 1 for the second. This project feeds single sentences, so the tensor is all zeros but still required by the Hub signature.

Inside the Hub layer, BERT combines the token embeddings with learned positional and segment embeddings, passes them through the encoder stack, and returns:
- `pooled_output`: The `[CLS]` embedding filtered through a dense + tanh layer, suitable for classification heads.
- `sequence_output`: Token-level contextual embeddings (length `max_seq_len`) useful for token classification tasks.

Our fine-tuning head consumes the pooled output, applies dropout, then predicts the binary label with a sigmoid-activated dense layer.

## Model Architecture and Training Code

- **Inputs**: three `tf.keras.Input` tensors (`input_word_ids`, `input_mask`, `input_type_ids`) with shape `(None, 128)`.
- **Encoder**: TensorFlow Hub BERT returns `pooled_output` (CLS embedding) and `sequence_output` (full token embeddings).
- **Head**: dropout layer (rate 0.4) followed by a dense classifier with sigmoid activation for binary labels.
- **Parameter count**: `109,483,010` trainable parameters (~417.6 MB) reported by `model.summary()`.
- **Compilation**: Adam optimizer with `learning_rate=2e-5`, `BinaryCrossentropy` loss, and `BinaryAccuracy` metric.
- **Training loop**: `model.fit` for 4 epochs using the `train_data`/`val_data` datasets above.

Excerpt of the training cell:
```python
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)
model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=4,
    verbose=1,
)
```

## GPU and Runtime Notes

- **GPU**: NVIDIA GeForce RTX 3080 Laptop GPU (16 GB) as reported by `nvidia-smi`.
- **Execution time**: first epoch ~142 s (graph compilation overhead), epochs 2–4 ~103–104 s each. Total training time ≈ 7.5 minutes for the 4-epoch run.
- **Batch size**: 32 samples per step (`drop_remainder=True`).
- If your system lacks the CUDA `libdevice` bundle, disable XLA as described in the setup section to avoid `libdevice.10.bc` errors during training.

## Results and Monitoring

- Validation accuracy peaked at ~95.5% across the run.
- Validation loss reached a minimum near 0.126 before slowly rising, indicating mild overfitting after epoch 2.
- Track training curves with the helper defined at the end of the notebook:
  ```python
  plot_graphs(history, "binary_accuracy")
  plot_graphs(history, "loss")
  ```
  (Requires `matplotlib.pyplot` to be imported and inline plotting enabled in the notebook.)

## Next Steps

- Increase the training sample size or apply regularization/early stopping for better generalization.
- Export the fine-tuned encoder via `model.save` or `tf.saved_model.save` for downstream deployment.
- Evaluate on the full Kaggle test set or other toxicity datasets to benchmark robustness.
