import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.interpolate import interp1d

def compute_voice_onset_advanced(file_path, nSD=3, minTH=0.15, 
                                 noise_duration=0.05, min_duration=0.02):
    """
    Computes voice onset using a hybrid threshold with proper noise floor estimation.
    
    Parameters:
    -----------
    file_path : str
        Path to audio file
    nSD : float
        Number of standard deviations above noise floor (default: 3)
    minTH : float
        Minimum threshold as fraction of peak (default: 0.15, i.e., 15%)
    noise_duration : float
        Duration (in seconds) to use for noise estimation (default: 0.05)
    min_duration : float
        Minimum duration above threshold to confirm onset (default: 0.02)
    """
    # 1. Load & Basic Cleanup
    y, sr = librosa.load(file_path, sr=None)
    y = y - np.mean(y)
    
    # 2. Add 50ms silence (as per VOAT step 2)
    silence_len = int(0.05 * sr)
    y_padded = np.concatenate([np.zeros(silence_len), y])
    
    # 3. Compute RMS Envelope (25ms window, 10ms step)
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    win = hann(frame_length)
    
    rms_env = []
    times_rms = []
    for i in range(0, len(y_padded) - frame_length, hop_length):
        rms_val = np.sqrt(np.mean((y_padded[i : i + frame_length] * win)**2))
        rms_env.append(rms_val)
        times_rms.append((i + frame_length // 2) / sr)
    
    rms_env = np.array(rms_env)
    times_rms = np.array(times_rms)
    
    # Interpolate to match signal resolution
    interp_func = interp1d(times_rms, rms_env, bounds_error=False, fill_value="extrapolate")
    full_times = np.arange(len(y_padded)) / sr
    env_full = interp_func(full_times)
    
    # --- CRITICAL FIX: Estimate noise from KNOWN quiet region ---
    # Use the first 50ms (the added silence + original start)
    noise_samples = int(noise_duration * sr)
    noise_region = env_full[:noise_samples]
    
    # Calculate Statistical Threshold from NOISE ONLY
    noise_mean = np.mean(noise_region)
    noise_std = np.std(noise_region)
    threshold_nSD = noise_mean + (noise_std * nSD)
    
    # Calculate Peak Threshold from ENTIRE signal
    peak_loudness = np.max(env_full)
    threshold_minTH = minTH * peak_loudness
    
    # Final Sound Threshold: The MAXIMUM of the two
    sound_threshold = max(threshold_nSD, threshold_minTH)
    
    # Convert to trial time (relative to actual signal start)
    trial_times = full_times - 0.05
    
    # --- Find onset with sustained crossing ---
    # Only search after 100ms to avoid pre-speech artifacts
    search_start_idx = np.where(trial_times >= 0.1)[0][0]
    
    # Find first crossing
    crossings = np.where(env_full[search_start_idx:] > sound_threshold)[0]
    
    if len(crossings) == 0:
        print(f"No threshold crossings found. Threshold: {sound_threshold:.6f}, Peak: {peak_loudness:.6f}")
        return None
    
    # Verify sustained crossing (signal stays above threshold)
    min_frames = int(min_duration * sr)
    
    for i, cross_idx in enumerate(crossings):
        actual_idx = cross_idx + search_start_idx
        # Check if signal stays above threshold for min_duration
        end_idx = min(actual_idx + min_frames, len(env_full))
        if np.all(env_full[actual_idx:end_idx] > sound_threshold):
            idx_onset = actual_idx
            break
    else:
        # No sustained crossing found
        print(f"No sustained crossing found (need {min_duration*1000:.0f}ms above threshold)")
        return None
    
    # Find saturation point (90% of peak)
    sat_threshold = 0.9 * peak_loudness
    idx_sat_candidates = np.where(env_full >= sat_threshold)[0]
    
    if len(idx_sat_candidates) > 0:
        idx_sat = idx_sat_candidates[0]
    else:
        # If no saturation, use the peak location
        idx_sat = np.argmax(env_full)
    
    # Convert back to signal time (remove padding offset)
    t_onset = (idx_onset - silence_len) / sr
    t_sat = (idx_sat - silence_len) / sr
    
    return {
        "t_onset": t_onset,
        "t_sat": t_sat,
        "threshold": sound_threshold,
        "threshold_nSD": threshold_nSD,
        "threshold_minTH": threshold_minTH,
        "peak": peak_loudness,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "times": full_times[silence_len:] - 0.05,
        "signal": y,
        "envelope": env_full[silence_len:]
    }

# --- Main Loop ---
input_dir = r"C:\Users\cletesson\OneDrive - UCL\Documents\test_voicekey"
files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
all_results = []

for f in files:
    print(f"\nProcessing: {f}")
    res = compute_voice_onset_advanced(os.path.join(input_dir, f), nSD=4, minTH=0.2)
    
    if res:
        all_results.append({
            "File": f,
            "Onset_Sec": res["t_onset"],
            "Saturation_Sec": res["t_sat"],
            "Threshold_Used": res["threshold"],
            "Threshold_nSD": res["threshold_nSD"],
            "Threshold_minTH": res["threshold_minTH"],
            "Peak_Loudness": res["peak"],
            "Noise_Mean": res["noise_mean"],
            "Noise_SD": res["noise_std"]
        })
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(res["times"], res["signal"], color='lightgray', alpha=0.5, label="Signal")
        plt.plot(res["times"], res["envelope"], color='blue', linewidth=2, label="RMS Envelope")
        plt.axhline(res["threshold"], color='orange', linestyle='--', linewidth=2, 
                   label=f"Hybrid Threshold ({res['threshold']:.6f})")
        plt.axhline(res["threshold_nSD"], color='purple', linestyle=':', 
                   label=f"Noise+{4}SD ({res['threshold_nSD']:.6f})")
        plt.axhline(res["threshold_minTH"], color='brown', linestyle=':', 
                   label=f"20% Peak ({res['threshold_minTH']:.6f})")
        plt.axhline(res["noise_mean"], color='red', linestyle=':', alpha=0.5,
                   label=f"Noise Mean ({res['noise_mean']:.6f})")
        plt.axvline(res["t_onset"], color='green', linewidth=2, 
                   label=f"Onset ({res['t_onset']*1000:.1f}ms)")
        plt.axvline(res["t_sat"], color='red', linewidth=2, 
                   label=f"Saturation ({res['t_sat']*1000:.1f}ms)")
        plt.axvline(0.1, color='gray', linestyle='--', alpha=0.5, label="Search Start (100ms)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Voice Onset Detection: {f}")
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print(f"  -> No valid onset detected")

# Export Results
if all_results:
    df = pd.DataFrame(all_results)
    output_path = os.path.join(input_dir, "onset_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print("Batch processing complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
else:
    print("\nNo valid onsets detected in any files.")