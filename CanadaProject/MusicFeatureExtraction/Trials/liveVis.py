
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import librosa.display
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample)
from PIL import Image

import pyaudio


from funcs import get_pitches, get_gradient

# Init Params
form = 8
channels = 2
rate = 44100
CHUNK = 2**11
hop_length = CHUNK//2
pa = pyaudio.PyAudio()
stream = pa.open(format=form,
                channels=channels,
                rate=rate,
                output=True)


# input shape = [?,100]
# generator = tf.keras.models.load_model("../../Networks/generator.hdf5")

model_name = "biggan-deep-128"
model = BigGAN.from_pretrained("biggan-deep-128")


# loading the song
lib_song, sr = librosa.load("../Music/beethovenwav.wav")

grads = []
num_inbetweens = 20
n_mels = 128
mel = librosa.filters.mel(sr=sr, n_fft=CHUNK, n_mels=n_mels)
frames = []
num_classes = 12

noise_vector = truncated_noise_sample(truncation=1, batch_size=1)

# All in tensors
noise_vector = torch.from_numpy(noise_vector)

for i in range(len(lib_song)//CHUNK):
    if i % 10 == 0:
        inp_vec = np.random.normal(size=100)
        print(i, len(lib_song)//CHUNK)
    for sample in [lib_song[CHUNK*i:CHUNK*(i+1)]]:
        ft_sample = librosa.stft(sample, n_fft=CHUNK, hop_length=CHUNK+1)

        amp = np.abs(ft_sample)**2
        spec = mel.dot(amp)

        grad = get_gradient(np.squeeze(spec))
        chroma = get_pitches(np.squeeze(spec), sr)

        cv1 = np.zeros(1000)
        chromasort = np.argsort(np.mean(chroma, axis=1))[::-1]
        for pi, p in enumerate(chromasort[:num_classes]):
            cv1[pi] = p

        stream.write(sample)

        class_vectors = torch.Tensor(np.array([cv1]))
        """with torch.no_grad():
            img = (model(noise_vector*grad, class_vectors, 1).numpy()+1)/2 * 255
            print(img.shape)
            plottable = Image.fromarray(
                np.rollaxis(img[0], 0, 3).astype(np.uint8))

            plt.imshow(plottable)
            plt.show()"""


stream.stop_stream()
stream.close()
pa.terminate()
