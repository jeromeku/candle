# Candle Tensor Tools

#### Utilities for porting Huggingface pytorch transformer models to Candle.

- [x] Convert pickled pytorch model files to safetensors format model files and upload to `Huggingface Hub`
- [x] Check expected model structure of models on `HF Hub` once in safetensors format
- [ ] Automatically creates corresponding `VarBuilder` paths from corresponding `safetensors` weight map

**Installation**

Clone the repo then:

```python
pip install --editable .
```

**Usage**

See [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb)
