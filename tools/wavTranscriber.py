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
    def speaker_text(speaker, text_string=None):
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
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wavSplit.frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    segments = wavSplit.vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)

    return [segment for segment in segments], sample_rate, audio_length 

def write_wave(audio, wav_name, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(wav_name, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

def write_audio_segments(segments, output_path, input_file_name, sample_rate=16000):
    """Write audio wave segments as diarization output"""
    output_path = os.path.join(output_path, os.path.splitext(os.path.basename(input_file_name))[0])
    os.makedirs(output_path, exist_ok=True)
    pairs  = find_pair(segments)
    for i, segment in enumerate(segments):
        segment.speaker = pairs[segment.speaker]
        duration = len(segment.bytes)/sample_rate/2
        tq = tqdm(total=duration)
        speaker = chr(ord("A")+segment.speaker)
        tq.set_description('{}. Speaker {}'.format(i, speaker))
        output_wave = os.path.join(output_path, "{}_Speaker_{}_{:.2f}_sec.wav".format(i, speaker, duration))
        write_wave(segment.bytes, output_wave, sample_rate)
        tq.update(duration)
        tq.close() 
    
def write_stt(segments, transcript_file, aggressive=3, sample_rate=16000, silence_thresh=1):
    """Writes audio speakers's segments with google speech recognition to a text file
    """
    vad = webrtcvad.Vad(aggressive)
    client = speech.SpeechClient()
    config = types.RecognitionConfig(encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
                                     sample_rate_hertz=sample_rate,
                                     language_code='en-US') 
    with open(transcript_file, 'w') as f:
        pairs  = find_pair(segments)
        for i, segment in enumerate(segments):
            segment.speaker = pairs[segment.speaker]
            duration = len(segment.bytes)/sample_rate/2
            tq = tqdm(total=duration)
            speaker = chr(ord("A")+segment.speaker)
            tq.set_description('{}-> Speaker {}'.format(i, speaker))
            silence_flag = check_silence(segment.bytes, vad, sample_rate=sample_rate, frame_duration_ms=30, silence_thresh= silence_thresh)
            if not silence_flag:
                output = segment_to_text(client, config, segment, sample_rate)
                
            else:
                output= PrintFormat.speaker_text(speaker, text_string='...') 
            f.write("{}. {}".format(i, output))
            f.flush()   
            tq.update(duration)
            tq.close()

def segment_to_text(client, config, segment, sample_rate=16000): 
    speaker = chr(ord("A")+segment.speaker) 
    if len(segment.bytes)/sample_rate/2 <= 60: # check whethere segment is more than 1 minute or not
        audio = types.RecognitionAudio(content = segment.bytes)
        response = client.recognize(config, audio)
        if not response.results:
            text_strings = ".."
        else:
            text_strings = response.results[0].alternatives[0].transcript.capitalize() 
    else:
        split_audios = gen_bytes_with_limit(segment.bytes, sample_rate, time_limit= 60)
        text_strings = ""
        for split_audio in split_audios:
            audio = types.RecognitionAudio(content = split_audio)
            response = client.recognize(config, audio)
            text_string = "" if not response.results else response.results[0].alternatives[0].transcript.capitalize()
            text_strings += text_string
            
    text = PrintFormat.speaker_text(speaker, text_string = text_strings)                   
    return text

def gen_bytes_with_limit(audio, sample_rate, time_limit=60): 
    frame_byte_count = int(sample_rate * time_limit * 2)
    offset = 0 
    while offset + frame_byte_count -1 < len(audio):
        yield audio[offset:offset + frame_byte_count] 
        offset += frame_byte_count
    yield audio[offset:]  
    
def find_pair(segments):
    labels = [segment.speaker for segment in segments]
    _, index = np.unique(labels, return_index=True)
    pairs = {labels[v]:k for k,v in enumerate(sorted(index))}
    return pairs
    
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

 