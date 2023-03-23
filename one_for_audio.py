# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 23:57:56 2023

@author: lenovo
"""

import warnings
import os
import numpy as np
import wave
import librosa
import sys
sys.path.append('D:/python/python3.8/lib/site-packages')
import tensorflow._api.v2.compat.v1 as tf
from vggish import vggish_input as vggish_input
from vggish import vggish_params as vggish_params
from vggish import vggish_postprocess as vggish_postprocess
from vggish import vggish_slim as vggish_slim
from vggish import loupe_keras as lpk

warnings.filterwarnings("ignore")
sys.path.append('E:/大创/Depression-recognition/DepressionCollected')
tf.disable_v2_behavior()
tf.enable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
prefix = os.path.abspath(os.path.join("E:/大创/Depression-recognition", "."))
# Paths to downloaded VGGish files.
checkpoint_path = "E:/大创/Depression-recognition/vggish/vggishggish_model.ckp"
pca_params_path = "E:/大创/Depression-recognition/vggish/vggish_params.py"
cluster_size = 16

min_len = 100
max_len = -1


# In[]
def to_vggish_embedds(x, sr):
    # x为输入的音频，sr为sample_rate
    input_batch = vggish_input.waveform_to_examples(x, sr)
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: input_batch})

    # Postprocess the results to produce whitened quantized embeddings.
    pproc = vggish_postprocess.Postprocessor(pca_params_path)
    postprocessed_batch = pproc.postprocess(embedding_batch)

    return tf.cast(postprocessed_batch, dtype='float32')


def wav2vlad(wave_data, sr):
    global cluster_size
    signal = wave_data
    melspec = librosa.feature.melspectrogram(y=signal, n_mels=80,sr=sr).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    feature_size = melspec.shape[1]
    max_samples = melspec.shape[0]
    output_dim = cluster_size * 16
    feat = lpk.NetVLAD(feature_size=feature_size, max_samples=max_samples, \
                       cluster_size=cluster_size, output_dim=output_dim) \
        (tf.convert_to_tensor(melspec))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        r = feat.numpy()
    return r


def extract_features(number, audio_features, path):
    global max_len, min_len
    if not os.path.exists("E:/大创/Depression-recognition/EATD-Corpus/" + str(path) + str(number)):
        return
    positive_file = wave.open(
        "E:/大创/Depression-recognition/EATD-Corpus/" + str(path) + str(number) + "/positive_out.wav")
    sr1 = positive_file.getframerate()
    nframes1 = positive_file.getnframes()
    wave_data1 = np.frombuffer(positive_file.readframes(nframes1), dtype=np.short).astype(np.float)
    len1 = nframes1 / sr1

    neutral_file = wave.open("E:/大创/Depression-recognition/EATD-Corpus/" + str(path) + str(number) + "/neutral_out.wav")
    sr2 = neutral_file.getframerate()
    nframes2 = neutral_file.getnframes()
    wave_data2 = np.frombuffer(neutral_file.readframes(nframes2), dtype=np.short).astype(np.float)
    len2 = nframes2 / sr2

    negative_file = wave.open(
        "E:/大创/Depression-recognition/EATD-Corpus/" + str(path) + str(number) + "/negative_out.wav")
    sr3 = negative_file.getframerate()
    nframes3 = negative_file.getnframes()
    wave_data3 = np.frombuffer(negative_file.readframes(nframes3), dtype=np.short).astype(np.float)
    len3 = nframes3 / sr3

    for l in [len1, len2, len3]:
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l

    if wave_data1.shape[0] < 1:
        wave_data1 = np.array([1e-4] * sr1 * 5)
    if wave_data2.shape[0] < 1:
        wave_data2 = np.array([1e-4] * sr2 * 5)
    if wave_data3.shape[0] < 1:
        wave_data3 = np.array([1e-4] * sr3 * 5)
    audio_features.append([wav2vlad(wave_data1, sr1), wav2vlad(wave_data2, sr2), wav2vlad(wave_data3, sr3)])


# In[]
audio_features = []

for index in range(1):
    extract_features(index + 1, audio_features, "t_")

# In[]
print("Saving npz file locally...")
np.savez("E:/大创/Depression-recognition/reg_feature/feature.npz", audio_features)
