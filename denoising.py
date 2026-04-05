import librosa
import soundfile as sf
import numpy as np
import os
import argparse
import glob
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def process_file(input_file, output_dir, threshold=0.1, min_duration=0.5, pad=0.2, save_filtered=False):
    """
    Extracts segments of bird sounds from an audio file.
    """
    print(f"\n--- Processing: {input_file} ---")
    
    try:
        # Load audio
        y, sr = librosa.load(input_file, sr=None)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return []
    
    # Bandpass filter (2kHz - 10kHz)
    lowcut = 2000.0
    highcut = 10000.0
    highcut = min(highcut, sr/2 - 100)
    
    y_filtered = bandpass_filter(y, lowcut, highcut, sr)
    
    # Calculate energy (RMS)
    frame_length = int(sr * 0.1)
    hop_length = int(sr * 0.05)
    rms = librosa.feature.rms(y=y_filtered, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize RMS
    if np.max(rms) > 0:
        rms = rms / np.max(rms)
    else:
        print("Silent file.")
        return []
    
    # Detect active frames
    active = rms > threshold
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    intervals = []
    start_time = None
    
    for i in range(len(active)):
        if active[i] and start_time is None:
            start_time = times[i]
        elif not active[i] and start_time is not None:
            end_time = times[i]
            if end_time - start_time >= min_duration:
                intervals.append((start_time, end_time))
            start_time = None
    
    if start_time is not None:
        end_time = times[-1]
        if end_time - start_time >= min_duration:
            intervals.append((start_time, end_time))
            
    if not intervals:
        print("No bird sounds detected.")
        return []
        
    # Merge segments within 0.5s
    merged = []
    curr_start, curr_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start - curr_end < 0.5:
            curr_end = next_end
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    
    # Save segments
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    file_output_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)
        
    extracted_files = []
    for i, (start, end) in enumerate(merged):
        start_pad = max(0, start - pad)
        end_pad = min(len(y)/sr, end + pad)
        
        start_idx = int(start_pad * sr)
        end_idx = int(end_pad * sr)
        
        if save_filtered:
            segment = y_filtered[start_idx:end_idx]
        else:
            segment = y[start_idx:end_idx]
        output_filename = f"{base_name}_segment_{i:03d}.wav"
        output_path = os.path.join(file_output_dir, output_filename)
        
        sf.write(output_path, segment, sr)
        print(f"  [+] Saved segment {i}: {start_pad:.2f}s - {end_pad:.2f}s (Duration: {end_pad-start_pad:.2f}s)")
        extracted_files.append(output_path)
        
    print(f"--- Finished: Extracted {len(merged)} segments ---")
    return extracted_files

def main():
    parser = argparse.ArgumentParser(description="Bird Sound Extraction Pipeline")
    parser.add_argument("input", help="Input .ogg file or directory containing .ogg files")
    parser.add_argument("--out", default="extracted_birds", help="Output root directory")
    parser.add_argument("--threshold", type=float, default=0.1, help="Energy threshold (0.0 to 1.0, default: 0.1)")
    parser.add_argument("--min_dur", type=float, default=0.5, help="Min duration in seconds (default: 0.5)")
    parser.add_argument("--pad", type=float, default=0.2, help="Padding in seconds (default: 0.2)")
    parser.add_argument("--filtered", action="store_true", help="Save filtered audio (2kHz-10kHz) instead of original")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.ogg"))
    else:
        files = [args.input]
        
    if not files:
        print("No .ogg files found.")
        return
        
    for f in files:
        process_file(f, args.out, args.threshold, args.min_dur, args.pad, args.filtered)

if __name__ == "__main__":
    main()