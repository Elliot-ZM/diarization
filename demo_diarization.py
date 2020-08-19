import os
import argparse
from tools import wavTranscriber 
import diarization
from timeit import default_timer as timer
from tqdm import tqdm 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(args):
    start = timer() 
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True) 
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(args.audio_file,
                                                                               aggressiveness=3,
                                                                               frame_duration_ms=30,
                                                                               padding_duration_ms=args.pad_silence_ms)
    segments = diarization.diarize(args, segments, 
                                   embedding_per_sec=1,
                                   overlap_rate=0.1)
    joined_segments = wavTranscriber.arrange_segments(segments)
        
    if args.opt == 'text':
        transcript_file = os.path.join(output_dir, os.path.basename(args.audio_file)[:-4] + '_transcript.txt')
        wavTranscriber.write_stt(joined_segments,
                                 transcript_file, 
                                 aggressive=3,
                                 sample_rate=sample_rate, 
                                 silence_thresh = args.silence_thresh)
    
    elif args.opt == 'audio':
        tq = tqdm(enumerate(joined_segments)) 
        for i, segment in tq:
            tq.set_description()
            wave_name = os.path.join(output_dir, '{}_{}_{}.wav'.format(os.path.basename(args.audio_file)[:-4],
                                                                       i, chr(ord("A")+segment.speaker)))
            wavTranscriber.write_wave(audio=segment.bytes,
                                      wav_name=wave_name,
                                      sample_rate=sample_rate) 
    end = timer() - start
    print("\nFinished in {:.2f} minute(s)".format(end/60))
    
    return segments, joined_segments

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', default="Mark Zuckerberg's 2004 Interview See How Far He And Facebook Have Come - YouTube.wav",
                        help='input audio file for diarization [mp3 or wav]')
    parser.add_argument('--num_speakers', type=int, default=4, 
                        help='manual speaker limit')
    parser.add_argument('--silence_thresh', type=int, default=1, 
                        help='remove silence speaker segment with given threshold, default "1 second"')
    parser.add_argument('--pad_silence_ms', type=int, default = 150,
                        help='pad silence duration in millisecond for each segment during voice activity detection')
    parser.add_argument('--opt', choices = ['text', 'audio'], default = 'audio',
                        help='option mode for output result')
    
    args = parser.parse_args()
    audio_path = r'/home/zmh/hdd/Custom_Projects/Speaker-Diarization/test-data'
    args.audio_file = os.path.join(audio_path, args.audio_file)     
    print(args)
    segments, joined_segments = main(args)
    wavTranscriber.PrintFormat.show_segments_info(segments)
    print()
    wavTranscriber.PrintFormat.show_segments_info(joined_segments)
    
        
        
    
    
    
    