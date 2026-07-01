"""
Regression tests for voice_onset_core.py.

These tests lock in a set of bug fixes made when the VOAT-style voice
onset/offset detection logic was extracted from a Tkinter GUI into a
pure, importable module:

1. compute_rms_envelope no longer flat-extrapolates the tail of the
   signal; trailing frames are shortened to fit instead.
2. compute_voice_onset_voat and detect_speech_boundaries_manual now
   agree: if search_delay exceeds the signal duration, BOTH report
   "no detection" rather than manual silently restarting at sample 0.
3. detect_speech_boundaries_manual now removes the DC offset the same
   way compute_voice_onset_voat does, so thresholding is bias-invariant.
4. load_wav correctly handles uint8 PCM (subtract 128, divide by 128).
5. Signals shorter than one analysis frame do not crash any of the
   detection functions.
6. Pure noise (no sustained energy) yields no detection.
7. find_sustained_crossing basic correctness for sustained vs.
   too-short crossings.

Tolerances are generous throughout since this is DSP, not exact
arithmetic.
"""

import numpy as np
import pytest
from scipy.io import wavfile

from voice_onset_core import (
    load_wav,
    compute_rms_envelope,
    find_sustained_crossing,
    compute_voice_onset_voat,
    detect_speech_boundaries_manual,
)


SR = 16000


def make_tone(duration_sec, sample_rate=SR, freq=200.0, amplitude=1.0, phase=0.0):
    n = int(duration_sec * sample_rate)
    t = np.arange(n) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def make_tone_burst(total_sec, onset_sec, offset_sec, sample_rate=SR,
                     freq=200.0, amplitude=1.0, noise_amp=0.001, dc_offset=0.0):
    """Near-silence (low-amplitude noise floor) except for a loud tone
    burst between onset_sec and offset_sec. Returns (signal, onset_idx_hint,
    offset_idx_hint) where the hints are sample indices of true onset/offset."""
    n = int(total_sec * sample_rate)
    rng = np.random.default_rng(42)
    sig = rng.normal(0, noise_amp, n).astype(np.float64)
    t = np.arange(n) / sample_rate
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    onset_idx = int(onset_sec * sample_rate)
    offset_idx = int(offset_sec * sample_rate)
    sig[onset_idx:offset_idx] += tone[onset_idx:offset_idx]
    sig += dc_offset
    return sig, onset_idx, offset_idx


# ---------------------------------------------------------------------------
# 1. compute_rms_envelope: tail must reflect real energy, not a flat carry.
# ---------------------------------------------------------------------------

def test_rms_envelope_tail_reflects_real_energy_not_flat_extrapolation():
    total_sec = 1.0
    n = int(total_sec * SR)
    # Loud tone for first 90%, near-silence for last 10%.
    quiet_start_sec = 0.9 * total_sec
    quiet_start_idx = int(quiet_start_sec * SR)

    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * 220.0 * t).astype(np.float64)
    rng = np.random.default_rng(0)
    sig[quiet_start_idx:] = rng.normal(0, 1e-4, n - quiet_start_idx)

    env, times = compute_rms_envelope(sig, SR, frame_length_ms=25, hop_length_ms=10)

    assert len(env) == n

    # Mean envelope value at ~80% through the signal (still loud tone region).
    idx_80 = int(0.80 * total_sec * SR)
    window_80 = int(0.02 * SR)  # 20ms window around 80%
    loud_region_mean = np.mean(env[idx_80:idx_80 + window_80])

    # Mean envelope value in the final 20ms of the signal (quiet region).
    final_20ms_samples = int(0.020 * SR)
    tail_mean = np.mean(env[-final_20ms_samples:])

    assert tail_mean < loud_region_mean * 0.5, (
        f"Tail envelope mean ({tail_mean}) should be meaningfully lower than "
        f"loud-region envelope mean ({loud_region_mean}); if it isn't, the "
        f"tail is likely being flat-extrapolated from an earlier frame "
        f"instead of reflecting the real (quiet) energy near the end."
    )


def test_rms_envelope_basic_shape_and_length():
    sig = make_tone(0.5)
    env, times = compute_rms_envelope(sig, SR, frame_length_ms=25, hop_length_ms=10)
    assert len(env) == len(sig)
    assert len(times) == len(sig)
    assert np.all(env >= 0)


# ---------------------------------------------------------------------------
# 2. search_delay longer than signal duration -> both functions report
#    "no detection".
# ---------------------------------------------------------------------------

def test_adaptive_no_detection_when_search_delay_exceeds_duration():
    sig, _, _ = make_tone_burst(total_sec=0.5, onset_sec=0.1, offset_sec=0.3)

    result = compute_voice_onset_voat(
        sig, SR,
        nSD=3.0, minTH=0.05, min_duration_ms=30,
        noise_window_ms=50, search_delay_ms=10_000,  # way beyond 0.5s duration
        frame_length_ms=25, hop_length_ms=10,
        offset_min_duration_ms=30,
    )

    assert result is None


def test_manual_no_detection_when_search_delay_exceeds_duration():
    sig, _, _ = make_tone_burst(total_sec=0.5, onset_sec=0.1, offset_sec=0.3)

    onset_idx, offset_idx, envelope, time_axis = detect_speech_boundaries_manual(
        sig, SR,
        manual_onset_threshold=0.1, manual_offset_threshold=0.1,
        manual_search_delay_ms=10_000,  # way beyond 0.5s duration
        min_duration_ms=30, offset_min_duration_ms=30,
        frame_length_ms=25, hop_length_ms=10,
    )

    assert onset_idx is None
    assert offset_idx is None
    # Envelope/time_axis are still returned (not silently reset to a search
    # from sample 0), consistent with the adaptive function's "no detection".
    assert len(envelope) == len(sig)
    assert len(time_axis) == len(sig)


# ---------------------------------------------------------------------------
# 3. DC-offset removal: both functions must be bias-invariant, and both
#    must agree with each other within ~1 hop length of the true onset/offset.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dc_offset", [0.0, 0.3, -0.4])
def test_dc_bias_invariant_detection_adaptive_and_manual(dc_offset):
    hop_ms = 10
    tolerance_sec = 2 * hop_ms / 1000  # generous: 2x hop length

    total_sec = 1.0
    onset_sec = 0.3
    offset_sec = 0.7

    sig, true_onset_idx, true_offset_idx = make_tone_burst(
        total_sec=total_sec, onset_sec=onset_sec, offset_sec=offset_sec,
        amplitude=1.0, noise_amp=0.001, dc_offset=dc_offset,
    )
    true_onset_sec = true_onset_idx / SR
    true_offset_sec = true_offset_idx / SR

    adaptive_result = compute_voice_onset_voat(
        sig, SR,
        nSD=3.0, minTH=0.05, min_duration_ms=30,
        noise_window_ms=100, search_delay_ms=0,
        frame_length_ms=25, hop_length_ms=hop_ms,
        offset_min_duration_ms=30,
    )
    assert adaptive_result is not None, f"adaptive detection failed for dc_offset={dc_offset}"
    assert abs(adaptive_result["t_onset"] - true_onset_sec) <= tolerance_sec
    assert abs(adaptive_result["t_offset"] - true_offset_sec) <= tolerance_sec

    manual_onset_idx, manual_offset_idx, envelope, time_axis = detect_speech_boundaries_manual(
        sig, SR,
        manual_onset_threshold=0.1, manual_offset_threshold=0.1,
        manual_search_delay_ms=0,
        min_duration_ms=30, offset_min_duration_ms=30,
        frame_length_ms=25, hop_length_ms=hop_ms,
    )
    assert manual_onset_idx is not None, f"manual detection failed for dc_offset={dc_offset}"
    assert manual_offset_idx is not None
    manual_onset_sec = time_axis[manual_onset_idx]
    manual_offset_sec = time_axis[manual_offset_idx]
    assert abs(manual_onset_sec - true_onset_sec) <= tolerance_sec
    assert abs(manual_offset_sec - true_offset_sec) <= tolerance_sec


def test_dc_bias_does_not_shift_detected_onset_offset_beyond_tolerance():
    hop_ms = 10
    tolerance_sec = 2 * hop_ms / 1000

    total_sec = 1.0
    onset_sec = 0.3
    offset_sec = 0.7

    sig_no_dc, _, _ = make_tone_burst(
        total_sec=total_sec, onset_sec=onset_sec, offset_sec=offset_sec,
        amplitude=1.0, noise_amp=0.001, dc_offset=0.0,
    )
    sig_dc, _, _ = make_tone_burst(
        total_sec=total_sec, onset_sec=onset_sec, offset_sec=offset_sec,
        amplitude=1.0, noise_amp=0.001, dc_offset=0.5,
    )

    kwargs = dict(
        nSD=3.0, minTH=0.05, min_duration_ms=30,
        noise_window_ms=100, search_delay_ms=0,
        frame_length_ms=25, hop_length_ms=hop_ms,
        offset_min_duration_ms=30,
    )
    result_no_dc = compute_voice_onset_voat(sig_no_dc, SR, **kwargs)
    result_dc = compute_voice_onset_voat(sig_dc, SR, **kwargs)

    assert result_no_dc is not None and result_dc is not None
    assert abs(result_no_dc["t_onset"] - result_dc["t_onset"]) <= tolerance_sec
    assert abs(result_no_dc["t_offset"] - result_dc["t_offset"]) <= tolerance_sec

    manual_kwargs = dict(
        manual_onset_threshold=0.1, manual_offset_threshold=0.1,
        manual_search_delay_ms=0,
        min_duration_ms=30, offset_min_duration_ms=30,
        frame_length_ms=25, hop_length_ms=hop_ms,
    )
    onset_no_dc, offset_no_dc, env_no_dc, time_no_dc = detect_speech_boundaries_manual(
        sig_no_dc, SR, **manual_kwargs
    )
    onset_dc, offset_dc, env_dc, time_dc = detect_speech_boundaries_manual(
        sig_dc, SR, **manual_kwargs
    )

    assert onset_no_dc is not None and onset_dc is not None
    assert offset_no_dc is not None and offset_dc is not None
    assert abs(time_no_dc[onset_no_dc] - time_dc[onset_dc]) <= tolerance_sec
    assert abs(time_no_dc[offset_no_dc] - time_dc[offset_dc]) <= tolerance_sec


# ---------------------------------------------------------------------------
# 4. load_wav: uint8 PCM handling.
# ---------------------------------------------------------------------------

def test_load_wav_uint8_pcm_is_centered_and_bounded(tmp_path):
    sr = SR
    t = np.arange(int(0.2 * sr)) / sr
    tone = np.sin(2 * np.pi * 300.0 * t)
    # Map [-1, 1] tone into uint8 PCM range [0, 255], centered at 128.
    uint8_data = (tone * 100 + 128).astype(np.uint8)

    wav_path = tmp_path / "uint8_test.wav"
    wavfile.write(str(wav_path), sr, uint8_data)

    loaded_sr, loaded_data, was_multichannel = load_wav(str(wav_path))

    assert loaded_sr == sr
    assert not was_multichannel
    assert loaded_data.dtype == np.float32

    # Centered near 0: the mean of a symmetric sine burst should be close to 0.
    assert abs(np.mean(loaded_data)) < 0.1

    # Roughly within [-1, 1] (allow tiny numerical slack).
    assert np.max(loaded_data) <= 1.0 + 1e-6
    assert np.min(loaded_data) >= -1.0 - 1e-6


def test_load_wav_uint8_pcm_full_range_maps_to_minus1_plus1(tmp_path):
    # 0 -> -1.0, 255 -> ~0.992, 128 -> 0.0
    uint8_data = np.array([0, 128, 255], dtype=np.uint8)
    wav_path = tmp_path / "uint8_range.wav"
    wavfile.write(str(wav_path), SR, uint8_data)

    _, loaded_data, _ = load_wav(str(wav_path))

    assert loaded_data[0] == pytest.approx(-1.0, abs=1e-6)
    assert loaded_data[1] == pytest.approx(0.0, abs=1e-6)
    assert loaded_data[2] == pytest.approx(127 / 128, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Robustness: signal shorter than one analysis frame must not crash.
# ---------------------------------------------------------------------------

def test_compute_rms_envelope_shorter_than_one_frame_does_not_crash():
    short_sig = np.random.default_rng(1).normal(0, 0.01, int(0.005 * SR))  # 5ms
    env, times = compute_rms_envelope(short_sig, SR, frame_length_ms=25, hop_length_ms=10)
    assert len(env) == len(short_sig)
    assert np.all(np.isfinite(env))


def test_compute_voice_onset_voat_shorter_than_one_frame_does_not_crash():
    short_sig = np.random.default_rng(2).normal(0, 0.01, int(0.005 * SR))  # 5ms
    result = compute_voice_onset_voat(
        short_sig, SR,
        nSD=3.0, minTH=0.05, min_duration_ms=30,
        noise_window_ms=50, search_delay_ms=0,
        frame_length_ms=25, hop_length_ms=10,
        offset_min_duration_ms=30,
    )
    # Too short to sustain any onset for min_duration_ms -> graceful None.
    assert result is None


def test_detect_speech_boundaries_manual_shorter_than_one_frame_does_not_crash():
    short_sig = np.random.default_rng(3).normal(0, 0.01, int(0.005 * SR))  # 5ms
    onset_idx, offset_idx, envelope, time_axis = detect_speech_boundaries_manual(
        short_sig, SR,
        manual_onset_threshold=0.1, manual_offset_threshold=0.1,
        manual_search_delay_ms=0,
        min_duration_ms=30, offset_min_duration_ms=30,
        frame_length_ms=25, hop_length_ms=10,
    )
    assert onset_idx is None
    assert offset_idx is None
    assert len(envelope) == len(short_sig)


def test_compute_rms_envelope_empty_signal_does_not_crash():
    env, times = compute_rms_envelope(np.array([]), SR, frame_length_ms=25, hop_length_ms=10)
    assert len(env) == 0
    assert len(times) == 0


# ---------------------------------------------------------------------------
# 6. Pure noise-only signal -> no detection from either function.
# ---------------------------------------------------------------------------

def test_adaptive_no_detection_on_pure_noise():
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.01, int(1.0 * SR))
    result = compute_voice_onset_voat(
        noise, SR,
        nSD=5.0, minTH=0.2, min_duration_ms=30,
        noise_window_ms=200, search_delay_ms=0,
        frame_length_ms=25, hop_length_ms=10,
        offset_min_duration_ms=30,
    )
    assert result is None


def test_manual_no_detection_on_pure_noise():
    rng = np.random.default_rng(8)
    noise = rng.normal(0, 0.01, int(1.0 * SR))
    onset_idx, offset_idx, envelope, time_axis = detect_speech_boundaries_manual(
        noise, SR,
        manual_onset_threshold=0.5,  # well above the noise floor
        manual_offset_threshold=0.5,
        manual_search_delay_ms=0,
        min_duration_ms=30, offset_min_duration_ms=30,
        frame_length_ms=25, hop_length_ms=10,
    )
    assert onset_idx is None
    assert offset_idx is None


# ---------------------------------------------------------------------------
# 7. find_sustained_crossing basic sanity.
# ---------------------------------------------------------------------------

def test_find_sustained_crossing_detects_sustained_rise():
    sample_rate = 1000
    n = 500
    envelope = np.zeros(n)
    step_idx = 100
    envelope[step_idx:] = 1.0  # rises and stays above threshold until the end

    min_duration_sec = 0.05  # 50 samples at 1000 Hz
    found = find_sustained_crossing(envelope, threshold=0.5, start_idx=0,
                                     min_duration_sec=min_duration_sec,
                                     sample_rate=sample_rate, direction='above')
    assert found == step_idx


def test_find_sustained_crossing_skips_too_short_rise_uses_later_sustained_one():
    sample_rate = 1000
    n = 500
    envelope = np.zeros(n)

    # A short blip above threshold, too brief to satisfy min_duration_sec.
    envelope[50:60] = 1.0  # only 10 samples above threshold

    # A later rise that is sufficiently sustained.
    sustained_start = 200
    envelope[sustained_start:] = 1.0

    min_duration_sec = 0.05  # 50 samples
    found = find_sustained_crossing(envelope, threshold=0.5, start_idx=0,
                                     min_duration_sec=min_duration_sec,
                                     sample_rate=sample_rate, direction='above')
    assert found == sustained_start


def test_find_sustained_crossing_returns_none_when_no_sustained_rise_exists():
    sample_rate = 1000
    n = 500
    envelope = np.zeros(n)

    # Several short blips, none long enough.
    envelope[50:60] = 1.0
    envelope[150:158] = 1.0
    envelope[400:405] = 1.0

    min_duration_sec = 0.05  # 50 samples
    found = find_sustained_crossing(envelope, threshold=0.5, start_idx=0,
                                     min_duration_sec=min_duration_sec,
                                     sample_rate=sample_rate, direction='above')
    assert found is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
