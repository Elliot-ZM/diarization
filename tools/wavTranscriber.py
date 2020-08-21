import webrtcvad
import os
from . import wavSplit
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums, types
from tqdm import tqdm
import contextlib
import wave
import numpy as np

class PrintFormat(object):
    def speaker_text(speaker, text_string=None, underline_text=True):
        if not text_string:
            return f'Speaker {speaker} : ""\n\n'
        return f'Speaker {speaker} : {text_string}.\n\n'
    
    def show_segments_info(segments):
        for seg in segments:
            print(f"Speaker: {seg.speaker}, audio length : {len(seg.bytes)/16000/2}, begin:end = {seg.begin:.2f} : {seg.end:.2f}")
        

class NewSegment(object):
    def __init__(self, bytes, begin, end, speaker):
        self.bytes = bytes
        self.begin = begin
        self.end = end
        self.speaker = speaker
 

def arrange_segments(segments):
    """
    Join same speaker segments if they are separted. 
    """
    new_segments = []
    new_idx = 0
    
    for cur_idx, cur_segment in enumerate(segments):
        if cur_idx==0:
            new_segments.append(NewSegment(cur_segment.bytes,
                                            cur_segment.begin,
                                            cur_segment.end, 
                                            cur_segment.speaker))
        elif cur_idx > 0:
            prev_segment = segments[cur_idx-1]
            change_flag = False if prev_segment.speaker==cur_segment.speaker else True # check speaker tag with previous and current segment
            if change_flag:
                new_segments.append(NewSegment(cur_segment.bytes,
                                               cur_segment.begin,
                                               cur_segment.end,
                                               cur_segment.speaker))
                new_idx = cur_idx
            else:
                new = new_segments[-1]
                new_audio = new.bytes + cur_segment.bytes
                new_segments[-1] = NewSegment(new_audio,
                                              segments[new_idx].begin,
                                              cur_segment.end,
                                              cur_segment.speaker)
    return new_segments

def vad_segment_generator(wavFile, aggressiveness, frame_duration_ms=30, padding_duration_ms=300):
    """Voice acitivity detection for speech recognition
    """
    wavFile = wavSplit.format_wave(wavFile) # formatting the input audio file for diarization
    audio, sample_rate, audio_length = wavSplit.read_wave(wavFile)
    assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wavSplit.frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    segments = wavSplit.vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)

    return [segment for segment in segments], sample_rate, audio_length

def segment_to_text(client, config, segment):
    audio = types.RecognitionAudio(content = segment.bytes)
    if len(segment.bytes)/16000/2 <= 60:
        response = client.recognize(config, audio)
    else:
        response = client.long_running_recognize(config, audio) ## changed with GCS url
    speaker = chr(ord("A")+segment.speaker)
    if not response.results:
        text = PrintFormat.speaker_text(speaker, underline_text=True)
    elif response.results:
        text_string = response.results[0].alternatives[0].transcript.capitalize()
        text = PrintFormat.speaker_text(speaker, text_string = text_string, underline_text= True)
                                             
    return text

def write_stt(segments, transcript_file, aggressive=3, sample_rate=16000, silence_thresh=1):
    """Writes audio speakers's segments with google speech recognition to a text file
    """
    vad = webrtcvad.Vad(aggressive)
    client = speech.SpeechClient()
    config = types.RecognitionConfig(encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
                                     sample_rate_hertz=sample_rate,
                                     language_code='en-US') 
    with open(transcript_file, 'w') as f:
        for i, segment in enumerate(segments):
            duration = len(segment.bytes)/sample_rate/2
            tq = tqdm(total=duration)
            speaker = chr(ord("A")+segment.speaker)
            tq.set_description('Speaker {}'.format(speaker))
            silence_flag = check_silence(segment.bytes, vad, sample_rate=sample_rate, frame_duration_ms=30, silence_thresh= silence_thresh)
            if not silence_flag:
                output = segment_to_text(client, config, segment)
                f.write(output)
                f.flush()
            else:
                wav_name = '{}_{}_{}.wav'.format(transcript_file[:-4].replace('transcript','non_voice'), i, speaker) 
                write_wave(segment.bytes, wav_name, sample_rate)
            tq.update(duration)
            tq.close() 
                
def write_wave(audio, wav_name, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(wav_name, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

def sort_speakers(labels):
    _, index = np.unique(labels, return_index=True)
    pairs = {labels[v]:k for k,v in enumerate(sorted(index))}
    sorted_labels = [pairs[label] for label in labels]
    return sorted_labels

def check_silence(audio, vad, sample_rate=16000, frame_duration_ms=30, silence_thresh=1):
    """Check voice duration of audio data and 
    return silence flag when the voiced duration is shorter than silence_thresh (second) 
    """
    frames = wavSplit.frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    voiced_frames = []
    for idx, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            voiced_frames.append(frame)

    voiced = b''.join([f.bytes for f in voiced_frames])
    voiced_duration = len(voiced)/sample_rate/2
    return False if voiced_duration >= silence_thresh else True

 