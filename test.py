from resemblyzer import preprocess_wav, VoiceEncoder
import os, sys

test_root = '/home/zmh/hdd/Custom_Projects/Speaker-Diarization/test-data'
audio_path = "Google's congressional hearing highlights in 11 minutes.wav"
wav_fpath = os.path.join(test_root, audio_path)

wav = preprocess_wav(wav_fpath)
encoder = VoiceEncoder("cpu")

_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=1.3)
print(cont_embeds.shape)

#%%
sys.path.append('..')
from tools import wavSplit
from spectralcluster import SpectralClusterer
import numpy as np
wav_fpath = wavSplit.format_wave(wav_fpath)

clusterer = SpectralClusterer(
    min_clusters=10, 
    max_clusters=100,
    p_percentile=0.90,
    gaussian_blur_sigma=1)

labels = clusterer.predict(cont_embeds)
print("There are '{}' Speakers ".format(np.unique(labels).shape[0]))
 
#%%
def create_labelling(labels, wav_splits):
    from resemblyzer import sampling_rate
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0
    
    for i, time in enumerate(times):
        if i>0 and labels[i] != labels[i-1]:
            temp = [str(labels[i-1]), start_time, time]
            labelling.append(tuple(temp))
            start_time = time
        if i==len(times)-1:
            temp = [str(labels[i]), start_time, time]
            labelling.append(tuple(temp))
            
    return labelling

labelling = create_labelling(labels, wav_splits)
print(labelling)
#%%
import math
import os
sample_rate = 16000
audio, sample_rate, audio_length = wavSplit.read_wave(wav_fpath) 

class NewSegment(object):
    def __init__(self, begin, end, speaker):
        self.begin = begin
        self.end = end
        self.speaker = speaker
 
def arrange_speaker(segments):
    """
    Join same speaker segments if they are separated. 
    """
    new_segments = []
    new_idx = 0
    for cur_idx, (turn, _ , speaker) in enumerate(segments):
        if cur_idx==0:
            new_segments.append(NewSegment(turn.start,
                                            turn.end, 
                                            speaker))
        elif cur_idx > 0:
            *_, prev_speaker = segments[cur_idx-1]
            change_flag = False if prev_speaker==speaker else True # check speaker tag with previous and current segment
            if change_flag:
                new_segments.append(NewSegment(turn.start,
                                               turn.end,
                                               speaker))
                new_idx = cur_idx
            else:
                new_turn, *_ = segments[new_idx]
                new_segments[-1] = NewSegment(new_turn.start,
                                              turn.end,
                                              speaker)
    return new_segments

def get_speaker_audio(audio, segment, sample_rate=16000):
    start_byte = int(sample_rate * math.floor(segment.begin) * 2)
    end_byte = int(sample_rate * math.ceil(segment.end) * 2)
    speaker_audio = audio[start_byte:end_byte]
    return speaker_audio
        
def gen_bytes_with_limit(audio, sample_rate, time_limit=60): 
    frame_byte_count = int(sample_rate * time_limit * 2)
    offset = 0 
    while offset + frame_byte_count -1 < len(audio):
        yield audio[offset:offset + frame_byte_count] 
        offset += frame_byte_count
    yield audio[offset:] 
    
