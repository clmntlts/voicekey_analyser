import numpy as np
from scipy import signal
from scipy.signal.windows import hann
from scipy.interpolate import interp1d
from scipy.io import wavfile


def load_wav(filepath):
    """Load a WAV file as mono float32 in [-1, 1]. Returns (sample_rate, data, was_multichannel)."""
    try:
        sample_rate, audio_raw = wavfile.read(filepath)
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier WAV '{filepath}': {e}") from e

    if audio_raw.dtype == np.int16:
        audio_data = audio_raw.astype(np.float32) / 32768.0
    elif audio_raw.dtype == np.int32:
        audio_data = audio_raw.astype(np.float32) / 2147483648.0
    elif audio_raw.dtype == np.uint8:
        audio_data = (audio_raw.astype(np.float32) - 128.0) / 128.0
    else:
        audio_data = audio_raw.astype(np.float32)

    was_multichannel = audio_data.ndim > 1
    if was_multichannel:
        audio_data = audio_data[:, 0]

    return sample_rate, audio_data, was_multichannel


def apply_highpass(data, sample_rate, cutoff_hz, enabled=True):
    """Zero-phase Butterworth high-pass filter, used to reduce low-frequency background noise."""
    if not enabled:
        return data

    nyquist = sample_rate / 2
    cutoff = cutoff_hz / nyquist
    if cutoff >= 1.0:
        cutoff = 0.99
    if cutoff <= 0.0:
        cutoff = 0.01

    b, a = signal.butter(4, cutoff, btype='high')
    return signal.filtfilt(b, a, data)


def compute_rms_envelope(data, sample_rate, frame_length_ms, hop_length_ms):
    """Hann-windowed RMS envelope. The final frame(s) are shortened to fit the
    signal instead of being left uncomputed, so the tail of the clip is backed
    by real energy rather than a flat extrapolated value."""
    n = len(data)
    full_times = np.arange(n) / sample_rate

    if n == 0:
        return np.array([]), full_times

    frame_length = max(1, int(frame_length_ms / 1000 * sample_rate))
    hop_length = max(1, int(hop_length_ms / 1000 * sample_rate))

    rms_env = []
    times_rms = []
    for i in range(0, n, hop_length):
        end = min(i + frame_length, n)
        segment = data[i:end]
        win = hann(len(segment)) if len(segment) > 1 else np.ones(len(segment))
        rms_env.append(np.sqrt(np.mean((segment * win) ** 2)))
        times_rms.append((i + end) / 2 / sample_rate)

    rms_env = np.array(rms_env)
    times_rms = np.array(times_rms)

    if len(times_rms) > 1:
        interp_func = interp1d(times_rms, rms_env, kind='linear',
                                bounds_error=False,
                                fill_value=(rms_env[0], rms_env[-1]))
        env_full = interp_func(full_times)
    else:
        env_full = np.full(n, rms_env[0] if len(rms_env) > 0 else 0.0)

    return env_full, full_times


def find_sustained_crossing(envelope, threshold, start_idx, min_duration_sec, sample_rate, direction='above'):
    """Find the first index where `envelope` stays above/below `threshold` for
    at least `min_duration_sec`. direction='above' searches forward for onset;
    direction='below' searches forward from onset for offset, falling back to
    the last sample above threshold if no fully-sustained drop is found."""
    min_samples = max(1, int(min_duration_sec * sample_rate))
    n = len(envelope)

    if direction == 'above':
        for i in range(start_idx, n):
            end_idx = min(i + min_samples, n)
            if np.all(envelope[i:end_idx] > threshold):
                return i
        return None
    else:
        for i in range(start_idx, n - min_samples + 1):
            if np.all(envelope[i:i + min_samples] < threshold):
                return i
        above = np.where(envelope[start_idx:] > threshold)[0]
        if len(above) > 0:
            return start_idx + above[-1]
        return start_idx


def compute_voice_onset_voat(data, sample_rate, *, nSD, minTH, min_duration_ms,
                              noise_window_ms, search_delay_ms, frame_length_ms,
                              hop_length_ms, offset_min_duration_ms):
    """Adaptive (VOAT-style) onset/offset detection. Returns None if no onset is found."""
    y = data - np.mean(data)
    env_full, full_times = compute_rms_envelope(y, sample_rate, frame_length_ms, hop_length_ms)

    if len(env_full) == 0:
        return None

    noise_samples = int(noise_window_ms / 1000 * sample_rate)
    if noise_samples > len(env_full):
        noise_samples = max(1, len(env_full) // 10)

    noise_region = env_full[:noise_samples]
    noise_mean = np.mean(noise_region)
    noise_std = np.std(noise_region)

    threshold_nSD = noise_mean + (noise_std * nSD)
    peak_loudness = np.max(env_full)
    threshold_minTH = minTH * peak_loudness
    sound_threshold = max(threshold_nSD, threshold_minTH)

    search_delay_sec = search_delay_ms / 1000
    search_start_idx = np.searchsorted(full_times, search_delay_sec)

    if search_start_idx >= len(env_full):
        return None

    min_onset_duration_sec = min_duration_ms / 1000
    onset_idx = find_sustained_crossing(env_full, sound_threshold, search_start_idx,
                                         min_onset_duration_sec, sample_rate, direction='above')

    if onset_idx is None:
        return None

    min_offset_duration_sec = offset_min_duration_ms / 1000
    offset_idx = find_sustained_crossing(env_full, sound_threshold, onset_idx,
                                          min_offset_duration_sec, sample_rate, direction='below')

    if offset_idx <= onset_idx:
        offset_idx = onset_idx + 1
    offset_idx = min(offset_idx, len(env_full) - 1)

    t_onset = full_times[onset_idx]
    t_offset = full_times[offset_idx]

    return {
        "t_onset": t_onset,
        "t_offset": t_offset,
        "onset_sample": onset_idx,
        "offset_sample": offset_idx,
        "threshold": sound_threshold,
        "threshold_nSD": threshold_nSD,
        "threshold_minTH": threshold_minTH,
        "peak": peak_loudness,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "envelope": env_full,
        "envelope_times": full_times
    }


def detect_speech_boundaries_manual(data, sample_rate, *, manual_onset_threshold,
                                     manual_offset_threshold, manual_search_delay_ms,
                                     min_duration_ms, offset_min_duration_ms,
                                     frame_length_ms, hop_length_ms):
    """Manual-threshold onset/offset detection. Kept symmetric with the adaptive
    function above: DC offset is removed the same way, and a search delay longer
    than the signal returns no detection instead of silently searching from 0."""
    y = data - np.mean(data)
    envelope, time_axis = compute_rms_envelope(y, sample_rate, frame_length_ms, hop_length_ms)

    if len(envelope) == 0:
        return None, None, envelope, time_axis

    search_delay_sec = manual_search_delay_ms / 1000
    search_start_idx = np.searchsorted(time_axis, search_delay_sec)

    if search_start_idx >= len(envelope):
        return None, None, envelope, time_axis

    min_onset_duration_sec = min_duration_ms / 1000
    onset_idx = find_sustained_crossing(envelope, manual_onset_threshold, search_start_idx,
                                         min_onset_duration_sec, sample_rate, direction='above')

    offset_idx = None
    if onset_idx is not None:
        min_offset_duration_sec = offset_min_duration_ms / 1000
        offset_idx = find_sustained_crossing(envelope, manual_offset_threshold, onset_idx,
                                              min_offset_duration_sec, sample_rate, direction='below')
        if offset_idx is not None and offset_idx <= onset_idx:
            offset_idx = onset_idx + 1

    return onset_idx, offset_idx, envelope, time_axis
