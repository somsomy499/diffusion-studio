"""Model quantization utilities."""
def quantize_model(model, bits=8):
    if bits == 8:
        return _int8_quantize(model)
    elif bits == 4:
        return _int4_quantize(model)
    raise ValueError(f"Unsupported bits: {bits}")

def _int8_quantize(model):
    return model

def _int4_quantize(model):
    return model
