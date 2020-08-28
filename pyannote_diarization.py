import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia_ami')

from pyannote.database import get_protocol
from pyannote.database import FileFinder
preprocessors = {'audio': FileFinder()}
protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset',
                        preprocessors=preprocessors)

test_file = next(protocol.test())

# test_file = {'uri' : 'filename', 'audio': audio_path}
diarization = pipeline(test_file)

for turn, _ , speaker in diarization.itertracks(yield_label=True):
    print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')


#%% Diarization Usage

import torch 
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia_ami')

audio_path = "/home/zmh/hdd/Custom_Projects/Speaker-Diarization/test-data/Google's congressional hearing highlights in 11 minutes.wav"
diarization = pipeline({'audio': audio_path})

# dump result to disk using RTTM format
with open('pyannote_result.rttm', 'w') as f:
    diarization.write_rttm(f)
    
# iterate over speech turns
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')
    
    
#%% check files length
from tools import wavSplit
from pydub import AudioSegment
from timeit import default_timer as timer
# for i, file in enumerate(protocol.test()):
#     data, sr, duration = wavSplit.read_wave(str(file['audio']))
#     print(f'{i}--> {duration}s, {sr}')
start  = timer()
test_file = list(protocol.test())[12]
diarization = pipeline(test_file)
for turn, _ , speaker in diarization.itertracks(yield_label=True):
    print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')

end = timer()
print(f"It take {end-start:.2f} seconds")
#%% 
start  = timer()
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
test_file = {'uri' : 'filename', 'audio': audio_path}
diarization = pipeline(test_file)

for turn, _ , speaker in diarization.itertracks(yield_label=True):
    print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')

end = timer()
print(f"It take {end-start:.2f} seconds")

#%% VISUALIZATION

from matplotlib import pyplot as plt
from pyannote.core import Segment, notebook
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
notebook.crop = Segment(0, 600)

# create a figure with 6 rows with matplotlib
nrows = 2
fig, ax = plt.subplots(nrows=nrows, ncols=1)
fig.set_figwidth(20)
fig.set_figheight(nrows *2)

# 1st row: reference annotation
# notebook.plot_annotation(test_file['annotation'], ax=ax[0], time=False)
# ax[0].text(notebook.crop.start + 0.5, 0.1, 'hypothesis', fontsize=14)

# 2nd row: pipeline output
notebook.plot_annotation(diarization, ax=ax[1], time=False)
ax[1].text(notebook.crop.start + 0.5, 0.1, 'hypothesis', fontsize=14)

#%%
from tools import wavTranscriber, wavSplit
import math
import os
sample_rate = 16000
audio, sample_rate, audio_length = wavSplit.read_wave(audio_path) 

class NewSegment(object):
    def __init__(self, begin, end, speaker):
        self.begin = begin
        self.end = end
        self.speaker = speaker
 
def arrange_speaker(diarization):
    """
    Join same speaker segments if they are separated. 
    """
    new_segments = []
    new_idx = 0
    segments = list(diarization.itertracks(yield_label=True))
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
    
joined_segments = arrange_speaker(diarization)
for seg in joined_segments:
    print(f"Speaker: {seg.speaker}, audio length : {seg.end-seg.begin:.2f}, begin:end = {seg.begin:.2f} : {seg.end:.2f}")
    seg.bytes = get_speaker_audio(audio, seg, sample_rate)
   
transcript_file = os.path.join(audio_path[:-4] + 'pyannote_dia.txt')
wavTranscriber.write_stt(joined_segments,
                        transcript_file, 
                        aggressive=3,
                        sample_rate=sample_rate, 
                        frame_duration_ms=30,
                        silence_thresh = 0.9)















