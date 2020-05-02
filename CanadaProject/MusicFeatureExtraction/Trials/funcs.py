import numpy as np
import librosa


def get_gradient(seq, clip=True):  # Have to figure out how to normalize if nec
    grad = np.gradient(seq)
    if clip:
        grad = grad.clip(min=0)
    return grad


def get_pitches(spec, sr):
    print("this is the spec", spec, "\n this is its shape", spec.shape)
    chroma = librosa.feature.chroma_stft(spec, sr)
    return chroma
