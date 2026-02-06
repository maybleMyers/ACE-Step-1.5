"""
Post-Processing Utilities Module
DSP functions for audio effects and processing
"""
import os
import tempfile
import numpy as np
import soundfile as sf
from scipy import signal
from pydub import AudioSegment


def save_audio_as_mp3(data, sr, is_stereo, metadata=None):
    """Save audio data as MP3 file with metadata and return the path."""
    temp_wav = tempfile.mktemp(suffix=".wav")
    if not is_stereo:
        data = data.flatten()
    sf.write(temp_wav, data, sr)

    output_path = tempfile.mktemp(suffix=".mp3")
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(output_path, format="mp3", bitrate="192k")
    os.remove(temp_wav)

    return output_path


def load_audio_for_processing(audio_path):
    """Load audio file and return sample rate, data, and metadata."""
    if audio_path is None:
        return None, None, "No audio file provided."

    try:
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            is_stereo = True
        else:
            is_stereo = False
            data = data.reshape(-1, 1)

        duration = len(data) / sr
        return (sr, data, is_stereo, None), None, f"Loaded: {os.path.basename(audio_path)} ({duration:.2f}s, {sr}Hz)"
    except Exception as e:
        return None, None, f"Error loading audio: {e}"


def trim_audio(audio_state, start_time, end_time):
    """Trim audio between two time points."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    duration = len(data) / sr

    if start_time < 0:
        start_time = 0
    if end_time > duration:
        end_time = duration
    if start_time >= end_time:
        return None, f"Invalid time range: start ({start_time:.2f}s) must be less than end ({end_time:.2f}s)"

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    trimmed_data = data[start_sample:end_sample]

    output_path = save_audio_as_mp3(trimmed_data, sr, is_stereo, metadata)
    new_duration = len(trimmed_data) / sr
    return output_path, f"Trimmed: {start_time:.2f}s to {end_time:.2f}s (new duration: {new_duration:.2f}s)"


def adjust_loudness(audio_state, gain_db):
    """Adjust audio loudness by a gain in dB."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    linear_gain = 10 ** (gain_db / 20)
    adjusted_data = data * linear_gain
    adjusted_data = np.clip(adjusted_data, -1.0, 1.0)

    output_path = save_audio_as_mp3(adjusted_data, sr, is_stereo, metadata)
    return output_path, f"Loudness adjusted by {gain_db:+.1f} dB"


def apply_bass_boost(audio_state, boost_db, cutoff_freq):
    """Apply bass boost using a low-shelf filter."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    original_rms = np.sqrt(np.mean(data ** 2))
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    b_low, a_low = signal.butter(2, normalized_cutoff, btype='low')
    bass_gain = 10 ** (boost_db / 20)

    result_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel = data[:, ch]
        bass = signal.filtfilt(b_low, a_low, channel)
        result_data[:, ch] = channel + bass * (bass_gain - 1)

    new_rms = np.sqrt(np.mean(result_data ** 2))
    if new_rms > 0:
        result_data = result_data * (original_rms / new_rms)
    result_data = np.tanh(result_data)

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    return output_path, f"Bass boost: {boost_db:+.1f} dB below {cutoff_freq} Hz"


def apply_treble_boost(audio_state, boost_db, cutoff_freq):
    """Apply treble boost using a high-shelf filter."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    original_rms = np.sqrt(np.mean(data ** 2))
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    b_high, a_high = signal.butter(2, normalized_cutoff, btype='high')
    treble_gain = 10 ** (boost_db / 20)

    result_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel = data[:, ch]
        treble = signal.filtfilt(b_high, a_high, channel)
        result_data[:, ch] = channel + treble * (treble_gain - 1)

    new_rms = np.sqrt(np.mean(result_data ** 2))
    if new_rms > 0:
        result_data = result_data * (original_rms / new_rms)
    result_data = np.tanh(result_data)

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    return output_path, f"Treble boost: {boost_db:+.1f} dB above {cutoff_freq} Hz"


def normalize_audio(audio_state, target_db):
    """Normalize audio to a target peak level in dB."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    current_peak = np.max(np.abs(data))
    if current_peak == 0:
        return None, "Audio is silent, cannot normalize."

    current_db = 20 * np.log10(current_peak)
    gain_db = target_db - current_db
    linear_gain = 10 ** (gain_db / 20)
    normalized_data = data * linear_gain

    output_path = save_audio_as_mp3(normalized_data, sr, is_stereo, metadata)
    return output_path, f"Normalized to {target_db:.1f} dB (gain: {gain_db:+.1f} dB)"


def apply_fade_in(audio_state, fade_duration):
    """Apply fade-in effect."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    fade_samples = int(fade_duration * sr)
    if fade_samples > len(data):
        fade_samples = len(data)

    fade_curve = np.linspace(0, 1, fade_samples)
    result_data = data.copy()
    for ch in range(data.shape[1]):
        result_data[:fade_samples, ch] = data[:fade_samples, ch] * fade_curve

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    return output_path, f"Fade-in applied: {fade_duration:.2f}s"


def apply_fade_out(audio_state, fade_duration):
    """Apply fade-out effect."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    fade_samples = int(fade_duration * sr)
    if fade_samples > len(data):
        fade_samples = len(data)

    fade_curve = np.linspace(1, 0, fade_samples)
    result_data = data.copy()
    for ch in range(data.shape[1]):
        result_data[-fade_samples:, ch] = data[-fade_samples:, ch] * fade_curve

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    return output_path, f"Fade-out applied: {fade_duration:.2f}s"


def change_speed(audio_state, speed_factor):
    """Change audio playback speed (affects pitch)."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    if speed_factor <= 0:
        return None, "Speed factor must be positive."

    new_length = int(len(data) / speed_factor)
    result_data = np.zeros((new_length, data.shape[1]))
    for ch in range(data.shape[1]):
        result_data[:, ch] = signal.resample(data[:, ch], new_length)

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    new_duration = new_length / sr
    return output_path, f"Speed changed to {speed_factor:.2f}x (new duration: {new_duration:.2f}s)"
