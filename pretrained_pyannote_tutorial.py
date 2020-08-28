import torch
# speech activity detection model trained on AMI training set
sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
# speaker change detection model trained on AMI training set
scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')
# overlapped speech detection model trained on AMI training set
ovl = torch.hub.load('pyannote/pyannote-audio', 'ovl_ami')
# speaker embedding model trained on AMI training set
emb = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')
# =============================================================================
# 
# #%% both device and batch size can be set manually
# sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami', device='cpu', batch_size=128)
# 
# #%%
# from pyannote.database import get_protocol
# from pyannote.database import FileFinder
# # preprocessors = {'audio': FileFinder()}
# # protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset',
# #                         preprocessors=preprocessors)
# # test_file = next(protocol.test())
# 
# #%% Segmentation (filename)
# audio_path = '/home/zmh/hdd/Custom_Projects/Speaker-Diarization/test-data/I am ... an expert.wav'
# test_file = {'uri': 'filename', 'audio': audio_path}
# 
# #%% Speech activity detection
# sad_scores = sad(test_file)
# 
# from pyannote.audio.utils.signal import Binarize
# binarize = Binarize(offset=0.52, onset=0.52, 
#                     log_scale=True,
#                     min_duration_off=0.1, min_duration_on=0.1)
# 
# # speech regions (as 'pyannote.core.Timeline' instance)
# speech = binarize.apply(sad_scores, dimension=1)
# 
# #%% Speaker change detection
# 
# scd_scores = scd(test_file)
# 
# from pyannote.audio.utils.signal import Peak
# peak = Peak(alpha=0.10, min_duration=0.10, log_scale=True)
# 
# # speaker change point (as `pyannote.core.Timeline` instance)
# partition = peak.apply(scd_scores, dimension=1)
# 
# =============================================================================
#%% Diarization

pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia_ami')

from pyannote.database import get_protocol
from pyannote.database import FileFinder

preprocessors = {'audio' : FileFinder}
protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset',
                        preprocessors=preprocessors)

test_file = next(protocol.test())

diarization = pipeline(test_file)









