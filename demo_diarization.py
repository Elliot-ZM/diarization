import os
import argparse
import wavTranscriber 
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums, types
import diarization
from timeit import default_timer as timer
from tqdm import tqdm
import webrtcvad

test_dir = r'/media/zmh/USB/test_data'

def main(args):
    start = timer()
    waveFile = os.path.join(test_dir, args.audio_file)
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile,
                                                                               aggressiveness=3,
                                                                               frame_duration_ms=30,
                                                                               padding_duration_ms=args.pad_silence_ms)
    segments = diarization.diarize(args, segments, embedding_per_sec=1.2, overlap_rate=0.4)
    joined_segments = wavTranscriber.arrange_segments(segments)
        
    if args.opt == 'text':
        transcript_file = os.path.join('results', 'test_transcript_' + os.path.basename(args.audio_file)[:-4] + '.txt')
        wavTranscriber.write_stt(joined_segments, transcript_file, aggressive=3, sample_rate=sample_rate, silence_thresh=args.silence_thresh)
    
    elif args.opt == 'audio':
        tq = tqdm(enumerate(joined_segments)) 
        for i, segment in tq:
            wavTranscriber.write_wave(os.path.join('results', f'{i}_{chr(ord("A")+segment.speaker)}_'+os.path.basename(args.audio_file)), segment.bytes, sample_rate)
    
    end = timer() - start
    print("\nFinished in {:.2f} minute(s)".format(end/60))
    
    return segments, joined_segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', default="Mark Zuckerberg's 2004 Interview See How Far He And Facebook Have Come - YouTube.MP3",
                        help='input audio file for diarization [mp3 or wav]')
    parser.add_argument('--num_speakers', type=int, default=4, 
                        help='manual speaker limit')
    parser.add_argument('--silence_thresh', type=int, default=1, 
                        help='remove silence speaker segment with given threshold, default "1 second"')
    parser.add_argument('--pad_silence_ms', type=int, default = 60,
                        help='pad silence duration in millisecond for each segment during voice activity detection')
    parser.add_argument('--opt', choices = ['text', 'audio'], default = 'text',
                        help='option mode for output result')
    
    args = parser.parse_args()
    segments, joined_segments = main(args)
    wavTranscriber.PrintFormat.show_segments_info(segments)
    print()
    wavTranscriber.PrintFormat.show_segments_info(joined_segments)
    
        
        
    
    
    
    