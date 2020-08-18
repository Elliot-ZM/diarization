import math
import numpy as np
import librosa
import uisrnn
import sys
sys.path.append('ghostvlad')
import model
from timeit import default_timer as timer

class Expando:
    def __init__(self, d):
        self.__d = d

    def __getattr__(self, item):
        return self.__d[item]


def get_magnitude(wav, win_len, hop_len):
    linear = librosa.stft(wav, n_fft=512, win_length=win_len, hop_length=hop_len)
    mag, phase = librosa.magphase(linear.T)
    return mag

def get_utterances_spec(mag, sr, hop_len, embedding_per_sec, overlap_rate):
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_len = sr / hop_len / embedding_per_sec
    spec_hop_len = spec_len * (1 - overlap_rate)
    cur_slide = 0.0
    utterances_spec = []

    while (True):
        if cur_slide + spec_len > time:
            break

        spec_mag = mag_T[:, int(cur_slide + 0.5):int(cur_slide + spec_len + 0.5)]
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)
        cur_slide += spec_hop_len

    return utterances_spec

def prepare_ghostvlad_data(segments, sr=16000, win_len=400, hop_len=160, embedding_per_sec=1.0, overlap_rate=0.1):
    active_wav = bytearray()
    for segment in segments:
        active_wav.extend(segment.bytes)
    mag = get_magnitude(np.array(librosa.util.buf_to_float(np.frombuffer(active_wav, dtype=np.int16))), win_len, hop_len)
    utterances_spec = get_utterances_spec(mag, sr, hop_len, embedding_per_sec, overlap_rate)

    return utterances_spec


def diarize(args, segments, sr=16000, win_len=400, hop_len=160, embedding_per_sec=1.0, overlap_rate=0.1):
    GHOSTVLAD_PATH = "ghostvlad/pretrained/weights.h5"
    UISRNN_PATH = "uisrnn/pretrained/saved_model.uisrnn_benchmark"
    
    start = timer()
    print("[INFO] Initializing dirization models")
    # Initialize ghostvlad
    ghostvlad_model = model.vggvox_resnet2d_icassp(input_dim=(257, None, 1),
                                                   num_class=5994,
                                                   mode="eval",
                                                   args=Expando({"net": "resnet34s",
                                                                 "loss": "softmax",
                                                                 "vlad_cluster": 8,
                                                                 "ghost_cluster": 2,
                                                                 "bottleneck_dim": 512,
                                                                 "aggregation_mode": "gvlad"}))
    ghostvlad_model.load_weights(GHOSTVLAD_PATH, by_name=True)

    # Initialize uisrnn
    sys.argv = sys.argv[:1]
    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    inference_args.num_speaker = args.num_speakers
    uisrnn_model = uisrnn.UISRNN(model_args)
    uisrnn_model.load(UISRNN_PATH)
    print("[INFO] Two models loading time : {:.2f} seconds.".format(timer()-start))
 
    # Extract D-vector with ghostvad
    print('[INFO] Extracting D-Vector from utterance.')
    utterances_spec = prepare_ghostvlad_data(segments, sr, win_len, hop_len, embedding_per_sec, overlap_rate)
    feats = []
    for spec in utterances_spec:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = ghostvlad_model.predict(spec)
        feats += [v]
    feats = np.array(feats)[:, 0, :].astype(float)

    # Clustering on d-vector with uisrnn
    labels = uisrnn_model.predict(feats, inference_args)
    print('[INFO] Tagging segments speakers.')
    embedding_duration = (1/embedding_per_sec) * (1.0 - overlap_rate)
    labels_count = len(labels)
    current = 0
    for segment in segments:
        begin_index = math.floor(current/embedding_duration)
        current += segment.end-segment.begin
        end_index = math.ceil(current/embedding_duration)
        segment_labels = [labels[index] for index in range(begin_index, min(end_index, labels_count))]
        if len(segment_labels) > 0:
            segment.speaker = max(segment_labels, key=segment_labels.count)
        else:
            segment.speaker = 999
    return segments
