"""
Post-Processing Utilities Module
DSP functions for audio effects and processing
"""
import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from scipy import signal
from pydub import AudioSegment
import torch
import gc

# Add the local AudioSR path for imports
_AUDIOSR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "versatile_audio_super_resolution"
)
if _AUDIOSR_PATH not in sys.path:
    sys.path.insert(0, _AUDIOSR_PATH)

# AudioSR model cache (lazy loaded to avoid startup delay)
_audiosr_model = None
_audiosr_model_name = None


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


# ============================================================================
# AudioSR Super-Resolution Functions
# ============================================================================

def get_audiosr_model(model_name="basic", device="auto"):
    """
    Get or create the AudioSR model (cached for reuse).

    Args:
        model_name: "basic" for general audio, "speech" for speech
        device: "auto", "cuda", "cpu", or "mps"

    Returns:
        AudioSR model instance
    """
    global _audiosr_model, _audiosr_model_name

    # Return cached model if same model requested
    if _audiosr_model is not None and _audiosr_model_name == model_name:
        return _audiosr_model

    # Unload previous model if different
    if _audiosr_model is not None:
        del _audiosr_model
        _audiosr_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Import and build model
    from audiosr import build_model

    # Handle device selection
    if device == "auto":
        device = None  # Let audiosr auto-detect

    _audiosr_model = build_model(model_name=model_name, device=device)
    _audiosr_model_name = model_name

    return _audiosr_model


def unload_audiosr_model():
    """Unload the AudioSR model to free memory."""
    global _audiosr_model, _audiosr_model_name

    if _audiosr_model is not None:
        del _audiosr_model
        _audiosr_model = None
        _audiosr_model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "AudioSR model unloaded successfully."
    return "No AudioSR model loaded."


def apply_lowpass_filter(audio_data, sr, cutoff_freq):
    """
    Apply low-pass filter to audio (preprocessing for AudioSR).

    Args:
        audio_data: numpy array of audio samples
        sr: sample rate
        cutoff_freq: cutoff frequency in Hz

    Returns:
        Filtered audio data
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    b, a = signal.butter(8, normalized_cutoff, btype='low')

    if len(audio_data.shape) == 1:
        return signal.filtfilt(b, a, audio_data)
    else:
        result = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[1]):
            result[:, ch] = signal.filtfilt(b, a, audio_data[:, ch])
        return result


def process_audio_chunk_audiosr(audiosr_model, chunk_path, ddim_steps, guidance_scale, seed):
    """
    Process a single audio chunk with AudioSR.

    Args:
        audiosr_model: AudioSR model instance
        chunk_path: path to temporary chunk file
        ddim_steps: number of DDIM steps
        guidance_scale: guidance scale for CFG
        seed: random seed

    Returns:
        Processed audio as numpy array
    """
    from audiosr import super_resolution

    waveform = super_resolution(
        audiosr_model,
        chunk_path,
        seed=seed,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
        latent_t_per_second=12.8
    )

    return waveform


def apply_audiosr(
    audio_state,
    model_name="basic",
    ddim_steps=50,
    guidance_scale=3.5,
    seed=-1,
    device="auto",
    apply_prefilter=True,
    prefilter_cutoff=12000,
    chunk_duration=5.1,
    overlap_duration=0.5,
    normalize_output=True,
    output_format="wav",
    progress_callback=None
):
    """
    Apply AudioSR super-resolution to audio.

    Args:
        audio_state: tuple of (sr, data, is_stereo, metadata)
        model_name: "basic" or "speech"
        ddim_steps: number of diffusion steps (10-200)
        guidance_scale: CFG guidance scale (1.0-20.0)
        seed: random seed (-1 for random)
        device: "auto", "cuda", "cpu"
        apply_prefilter: whether to apply low-pass filter before processing
        prefilter_cutoff: low-pass filter cutoff frequency
        chunk_duration: duration of each chunk in seconds
        overlap_duration: overlap between chunks for crossfade
        normalize_output: whether to normalize output audio
        output_format: "wav", "mp3", or "flac"
        progress_callback: optional callback for progress updates

    Returns:
        tuple of (output_path, status_message)
    """
    import random
    import librosa

    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state

    # Handle random seed
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    try:
        # Load the model
        if progress_callback:
            progress_callback("Loading AudioSR model...")

        audiosr_model = get_audiosr_model(model_name, device)

        # Prepare audio data (ensure float32, shape: [samples, channels])
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Ensure 2D array
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Apply pre-filter if requested (helps with compressed audio)
        if apply_prefilter:
            if progress_callback:
                progress_callback(f"Applying low-pass filter at {prefilter_cutoff}Hz...")
            data = apply_lowpass_filter(data, sr, prefilter_cutoff)

        # Resample to 48kHz if needed (AudioSR expects 48kHz)
        target_sr = 48000
        if sr != target_sr:
            if progress_callback:
                progress_callback(f"Resampling from {sr}Hz to {target_sr}Hz...")
            # Resample each channel
            resampled_channels = []
            for ch in range(data.shape[1]):
                resampled = librosa.resample(data[:, ch], orig_sr=sr, target_sr=target_sr)
                resampled_channels.append(resampled)
            data = np.stack(resampled_channels, axis=1)
            sr = target_sr

        # Calculate chunk parameters
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)
        step_samples = chunk_samples - overlap_samples
        total_samples = len(data)

        # Process each channel separately
        processed_channels = []
        num_channels = data.shape[1]

        for ch_idx in range(num_channels):
            if progress_callback:
                progress_callback(f"Processing channel {ch_idx + 1}/{num_channels}...")

            channel_data = data[:, ch_idx]
            processed_chunks = []

            # Calculate number of chunks
            num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step_samples)))

            for i in range(num_chunks):
                start = i * step_samples
                end = min(start + chunk_samples, total_samples)

                if progress_callback:
                    progress_callback(f"Channel {ch_idx + 1}: Processing chunk {i + 1}/{num_chunks}...")

                # Extract chunk
                chunk = channel_data[start:end]

                # Pad if chunk is too short
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

                # Save chunk to temp file
                temp_chunk_path = tempfile.mktemp(suffix=".wav")
                sf.write(temp_chunk_path, chunk, sr)

                try:
                    # Process chunk with consistent seed for uniform processing
                    processed_chunk = process_audio_chunk_audiosr(
                        audiosr_model,
                        temp_chunk_path,
                        ddim_steps,
                        guidance_scale,
                        seed  # Use same seed for all chunks for consistency
                    )

                    # Convert to numpy and squeeze
                    if isinstance(processed_chunk, torch.Tensor):
                        processed_chunk = processed_chunk.cpu().numpy()
                    processed_chunk = np.squeeze(processed_chunk)

                    # Ensure 1D
                    if len(processed_chunk.shape) > 1:
                        processed_chunk = processed_chunk[0]

                    # Trim padding from last chunk
                    actual_chunk_len = end - start
                    scale_factor = len(processed_chunk) / chunk_samples
                    target_len = int(actual_chunk_len * scale_factor)
                    processed_chunk = processed_chunk[:target_len]

                    processed_chunks.append(processed_chunk)

                finally:
                    # Clean up temp file
                    if os.path.exists(temp_chunk_path):
                        os.remove(temp_chunk_path)

            # Crossfade and concatenate chunks
            if progress_callback:
                progress_callback(f"Channel {ch_idx + 1}: Concatenating chunks...")

            if len(processed_chunks) == 1:
                channel_result = processed_chunks[0]
            else:
                # Calculate overlap in output samples
                output_sr = 48000  # AudioSR outputs at 48kHz
                output_overlap = int(overlap_duration * output_sr)

                # Crossfade chunks using equal-power (constant-power) curves
                channel_result = processed_chunks[0]
                for j in range(1, len(processed_chunks)):
                    next_chunk = processed_chunks[j].copy()  # Don't modify original

                    # Ensure we have enough samples for crossfade
                    actual_overlap = min(output_overlap, len(channel_result), len(next_chunk))

                    if actual_overlap > 0:
                        # Create equal-power crossfade curves (constant power throughout)
                        # Using cosine curves: fade_out² + fade_in² = 1 at all points
                        t = np.linspace(0, np.pi / 2, actual_overlap)
                        fade_out = np.cos(t)  # Goes from 1 to 0
                        fade_in = np.sin(t)   # Goes from 0 to 1

                        # Apply equal-power crossfade
                        crossfaded = (channel_result[-actual_overlap:] * fade_out +
                                     next_chunk[:actual_overlap] * fade_in)

                        # Replace the overlap region and append the rest
                        channel_result = np.concatenate([
                            channel_result[:-actual_overlap],
                            crossfaded,
                            next_chunk[actual_overlap:]
                        ])
                    else:
                        channel_result = np.concatenate([channel_result, next_chunk])

            processed_channels.append(channel_result)

        # Combine channels
        if progress_callback:
            progress_callback("Finalizing output...")

        # Ensure all channels have same length
        min_len = min(len(ch) for ch in processed_channels)
        processed_channels = [ch[:min_len] for ch in processed_channels]

        result_data = np.stack(processed_channels, axis=1)

        # Normalize if requested
        if normalize_output:
            max_val = np.max(np.abs(result_data))
            if max_val > 0:
                result_data = result_data / max_val * 0.95  # Leave some headroom

        # Clip to valid range
        result_data = np.clip(result_data, -1.0, 1.0)

        # Save output
        output_sr = 48000  # AudioSR always outputs at 48kHz

        if output_format == "mp3":
            output_path = save_audio_as_mp3(result_data, output_sr, is_stereo, metadata)
        elif output_format == "flac":
            output_path = tempfile.mktemp(suffix=".flac")
            if not is_stereo:
                result_data = result_data.flatten()
            sf.write(output_path, result_data, output_sr, format='FLAC')
        else:  # wav
            output_path = tempfile.mktemp(suffix=".wav")
            if not is_stereo:
                result_data = result_data.flatten()
            sf.write(output_path, result_data, output_sr)

        new_duration = len(result_data) / output_sr
        return output_path, f"✅ Super-resolution complete! Output: {output_sr}Hz, {new_duration:.2f}s (seed: {seed})"

    except Exception as e:
        import traceback
        error_msg = f"AudioSR failed: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def audiosr_super_resolution(
    audio_path,
    model_name,
    ddim_steps,
    guidance_scale,
    seed,
    device,
    apply_prefilter,
    prefilter_cutoff,
    chunk_duration,
    overlap_duration,
    normalize_output,
    output_format
):
    """
    Wrapper function for Gradio interface.
    Loads audio and applies AudioSR super-resolution.
    """
    if audio_path is None:
        return None, "No audio file provided."

    # Load the audio
    audio_state, _, load_status = load_audio_for_processing(audio_path)

    if audio_state is None:
        return None, load_status

    # Apply super-resolution
    return apply_audiosr(
        audio_state=audio_state,
        model_name=model_name,
        ddim_steps=int(ddim_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        device=device,
        apply_prefilter=apply_prefilter,
        prefilter_cutoff=int(prefilter_cutoff),
        chunk_duration=float(chunk_duration),
        overlap_duration=float(overlap_duration),
        normalize_output=normalize_output,
        output_format=output_format
    )
