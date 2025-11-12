"""
Baby Cry Audio Isolation and Filtering Module.
Implements advanced audio processing techniques to isolate baby cries from background noise.
"""

import os
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Tuple, List, Optional, Union
import scipy.signal
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings("ignore")

from .config import Config
from .data_preprocessing import AudioPreprocessor
from .model import create_model
from .acoustic_features import AcousticFeatureExtractor, validate_cry_with_acoustic_features
from .calibration import ConfidenceCalibrator


class BabyCryAudioFilter:
    """
    Advanced audio filtering system to isolate baby cries from mixed audio.
    Combines spectral filtering, voice activity detection, and deep learning classification.
    """

    def __init__(self, config: Config = Config(), model_path: Optional[str] = None,
                 calibrator_path: Optional[str] = None):
        """
        Initialize the baby cry audio filter.

        Args:
            config: Configuration object
            model_path: Path to trained baby cry detection model
            calibrator_path: Path to confidence calibrator (optional)
        """
        self.config = config
        self.sample_rate = config.SAMPLE_RATE
        self.preprocessor = AudioPreprocessor(config)

        # Baby cry frequency characteristics (Hz)
        self.cry_freq_min = 100    # Fundamental frequency minimum
        self.cry_freq_max = 3000   # Harmonic content maximum
        self.cry_peak_freq = 400   # Typical crying peak frequency

        # Load trained model for cry detection
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Initialize acoustic feature extractor
        self.acoustic_extractor = AcousticFeatureExtractor(sample_rate=self.sample_rate)

        # Initialize confidence calibrator
        self.calibrator = None
        if calibrator_path and os.path.exists(calibrator_path):
            self.calibrator = ConfidenceCalibrator(method="temperature")
            self.calibrator.load(calibrator_path, num_classes=config.NUM_CLASSES)
            print(f"Loaded calibrator from {calibrator_path}")

        # Initialize transforms
        self._init_transforms()

    def _init_transforms(self):
        """Initialize audio processing transforms."""
        # Spectral transforms
        self.stft_transform = T.Spectrogram(
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            power=2.0
        )

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            f_min=self.config.F_MIN,
            f_max=self.config.F_MAX
        )

        # Inverse transforms for reconstruction (more iterations = better quality)
        self.griffin_lim = T.GriffinLim(
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_iter=100  # Increased from 32 for much better audio quality
        )

    def load_model(self, model_path: str):
        """Load trained baby cry detection model."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if 'model' in checkpoint and not isinstance(checkpoint['model'], dict):
                # Quantized model format: checkpoint['model'] is the actual model object
                self.model = checkpoint['model']
            else:
                # Standard format: load state_dict into a fresh model
                self.model = create_model(self.config)

                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self.model = None

    def spectral_filter(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain filtering to emphasize baby cry frequencies.

        Args:
            audio: Input audio tensor

        Returns:
            Filtered audio tensor
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            return_complex=True
        )

        # Create frequency mask for baby cry range
        freqs = torch.fft.fftfreq(self.config.N_FFT, 1/self.sample_rate)[:self.config.N_FFT//2 + 1]

        # Frequency-based filter with smooth transitions (better audio quality)
        freq_mask = torch.ones_like(freqs) * 0.3  # Keep some background for naturalness

        # Smooth bandpass filter for cry frequencies
        cry_band = (freqs >= self.cry_freq_min) & (freqs <= self.cry_freq_max)
        freq_mask[cry_band] = 1.0

        # Gentle emphasis on typical cry frequencies (avoid over-amplification)
        peak_band = (freqs >= 300) & (freqs <= 600)
        freq_mask[peak_band] = 1.3  # Reduced from 2.0 to avoid distortion

        # Apply smooth roll-off at edges (reduces artifacts)
        transition_width = 50  # Hz
        for i, f in enumerate(freqs):
            if self.cry_freq_min - transition_width < f < self.cry_freq_min:
                # Smooth transition at low end
                alpha = (f - (self.cry_freq_min - transition_width)) / transition_width
                freq_mask[i] = 0.3 + alpha * 0.7
            elif self.cry_freq_max < f < self.cry_freq_max + transition_width:
                # Smooth transition at high end
                alpha = 1 - (f - self.cry_freq_max) / transition_width
                freq_mask[i] = 0.3 + alpha * 0.7

        # Apply filter
        stft_filtered = stft * freq_mask.unsqueeze(-1)

        # Reconstruct audio
        filtered_audio = torch.istft(
            stft_filtered,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            length=audio.shape[-1]
        )

        return filtered_audio

    def detect_harmonic_structure(self, audio: torch.Tensor,
                                 frame_length: int = 2048,
                                 hop_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect harmonic structure characteristic of baby cries.
        Baby cries have clear fundamental frequency + harmonic overtones.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Tuple of (harmonic_confidence_per_frame, fundamental_frequencies)
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)

        # Frequency bins
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]
        max_freq = freqs.max().item()  # Get actual maximum frequency (not freqs[-1] which can be negative)

        # Find fundamental frequency in baby cry range (300-600 Hz)
        cry_f0_min_idx = torch.argmin(torch.abs(freqs - 300))
        cry_f0_max_idx = torch.argmin(torch.abs(freqs - 600))

        # For each time frame, find peaks
        harmonic_scores = []
        f0_estimates = []

        for frame_idx in range(magnitude.shape[1]):
            frame_mag = magnitude[:, frame_idx]

            # Find fundamental frequency (strongest peak in 300-600 Hz)
            cry_range_mag = frame_mag[cry_f0_min_idx:cry_f0_max_idx]
            if cry_range_mag.max() < 1e-6:
                harmonic_scores.append(0.0)
                f0_estimates.append(0.0)
                continue

            local_peak_idx = torch.argmax(cry_range_mag)
            f0_idx = cry_f0_min_idx + local_peak_idx
            f0 = freqs[f0_idx].item()

            # Check for harmonics at 2*f0, 3*f0, 4*f0
            harmonic_strength = 0.0
            num_harmonics_found = 0

            for harmonic_num in [2, 3, 4]:
                expected_freq = f0 * harmonic_num
                if expected_freq > max_freq:
                    break

                # Find nearest frequency bin
                harmonic_idx = torch.argmin(torch.abs(freqs - expected_freq))

                # Check if there's a peak within +/-50 Hz tolerance
                tolerance_bins = int(50 / (self.sample_rate / frame_length))
                start_idx = max(0, harmonic_idx - tolerance_bins)
                end_idx = min(len(frame_mag), harmonic_idx + tolerance_bins)

                local_max = torch.max(frame_mag[start_idx:end_idx])
                f0_magnitude = frame_mag[f0_idx]

                # Harmonic should be weaker than fundamental but still present
                if local_max > 0.1 * f0_magnitude:
                    harmonic_strength += local_max / f0_magnitude
                    num_harmonics_found += 1

            # Score based on number and strength of harmonics
            if num_harmonics_found >= 2:
                score = min(1.0, harmonic_strength / 3.0)
            else:
                score = 0.0

            harmonic_scores.append(score)
            f0_estimates.append(f0)

        return torch.tensor(harmonic_scores), torch.tensor(f0_estimates)

    def detect_temporal_patterns(self, audio: torch.Tensor,
                                cry_burst_min: float = 0.3,
                                cry_burst_max: float = 2.0,
                                pause_min: float = 0.1,
                                pause_max: float = 0.8) -> torch.Tensor:
        """
        Detect temporal patterns: cry bursts followed by inhalation pauses.
        Baby cries have characteristic rhythm of vocalization + brief silence.

        Args:
            audio: Input audio tensor
            cry_burst_min: Minimum cry burst duration (seconds)
            cry_burst_max: Maximum cry burst duration (seconds)
            pause_min: Minimum pause duration (seconds)
            pause_max: Maximum pause duration (seconds)

        Returns:
            Temporal pattern confidence score per frame
        """
        # Compute short-term energy
        frame_length = int(0.05 * self.sample_rate)  # 50ms frames
        hop_length = int(0.025 * self.sample_rate)   # 25ms hop

        frames = audio.unfold(-1, frame_length, hop_length)
        energy = torch.mean(frames ** 2, dim=-1)

        # Normalize energy
        energy = (energy - energy.mean()) / (energy.std() + 1e-8)

        # Detect active vs pause segments
        energy_threshold = 0.0  # Above mean
        is_active = energy > energy_threshold

        # Find transitions
        transitions = torch.diff(is_active.float())
        burst_starts = torch.where(transitions > 0)[0]
        burst_ends = torch.where(transitions < 0)[0]

        # Align starts and ends
        if len(burst_starts) == 0 or len(burst_ends) == 0:
            return torch.zeros(len(energy))

        if burst_ends[0] < burst_starts[0]:
            burst_ends = burst_ends[1:]
        if len(burst_starts) > len(burst_ends):
            burst_starts = burst_starts[:len(burst_ends)]

        # Analyze burst durations
        pattern_scores = torch.zeros(len(energy))

        for i in range(len(burst_starts) - 1):
            burst_duration = (burst_ends[i] - burst_starts[i]) * hop_length / self.sample_rate
            pause_duration = (burst_starts[i+1] - burst_ends[i]) * hop_length / self.sample_rate

            # Check if matches cry pattern
            burst_match = (cry_burst_min <= burst_duration <= cry_burst_max)
            pause_match = (pause_min <= pause_duration <= pause_max)

            if burst_match and pause_match:
                # Mark this region with high score
                pattern_scores[burst_starts[i]:burst_ends[i]] = 1.0

        return pattern_scores

    def track_pitch_contours(self, audio: torch.Tensor,
                           frame_length: int = 2048,
                           hop_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track pitch contours to identify rising/falling patterns unique to infant distress.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Tuple of (contour_scores, pitch_track)
        """
        # Use autocorrelation for pitch tracking
        pitch_track = []

        for start_idx in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start_idx:start_idx + frame_length]

            # Autocorrelation method
            autocorr = torch.nn.functional.conv1d(
                frame.unsqueeze(0).unsqueeze(0),
                frame.flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame_length - 1
            )[0, 0, frame_length-1:]

            # Look for peaks in baby cry F0 range (300-600 Hz)
            min_lag = int(self.sample_rate / 600)  # Max freq -> min lag
            max_lag = int(self.sample_rate / 300)  # Min freq -> max lag

            if max_lag < len(autocorr):
                autocorr_range = autocorr[min_lag:max_lag]
                if autocorr_range.max() > 0:
                    peak_lag = min_lag + torch.argmax(autocorr_range).item()
                    pitch = self.sample_rate / peak_lag
                else:
                    pitch = 0.0
            else:
                pitch = 0.0

            pitch_track.append(pitch)

        pitch_track = torch.tensor(pitch_track)

        # Analyze pitch contours for cry-like patterns
        contour_scores = torch.zeros(len(pitch_track))

        # Look for rising/falling patterns typical of cries
        window_size = int(0.3 * self.sample_rate / hop_length)  # 300ms windows

        for i in range(window_size, len(pitch_track) - window_size):
            window = pitch_track[i-window_size:i+window_size]
            valid_pitches = window[window > 0]

            if len(valid_pitches) > window_size:
                # Calculate pitch variation
                pitch_std = valid_pitches.std().item()
                pitch_range = (valid_pitches.max() - valid_pitches.min()).item()

                # Baby cries have moderate pitch variation (not flat, not erratic)
                if 20 < pitch_range < 200 and pitch_std > 10:
                    contour_scores[i] = min(1.0, pitch_range / 150)

        return contour_scores, pitch_track

    def detect_frequency_modulation(self, audio: torch.Tensor,
                                   frame_length: int = 2048,
                                   hop_length: int = 512,
                                   modulation_rate_min: float = 3.0,
                                   modulation_rate_max: float = 12.0) -> torch.Tensor:
        """
        Detect rapid vibrato-like frequency modulation characteristic of baby cries.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length
            modulation_rate_min: Minimum FM rate in Hz
            modulation_rate_max: Maximum FM rate in Hz

        Returns:
            FM detection scores per frame
        """
        # Compute instantaneous frequency using STFT phase
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )

        # Phase unwrapping and differentiation for instantaneous frequency
        phase = torch.angle(stft)
        phase_diff = torch.diff(phase, dim=1)

        # Instantaneous frequency
        inst_freq = phase_diff * self.sample_rate / (2 * np.pi * hop_length)

        # For each frequency bin in cry range, detect modulation
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]
        cry_bins = (freqs >= 300) & (freqs <= 600)

        fm_scores = []

        for frame_idx in range(inst_freq.shape[1]):
            # Get instantaneous frequencies in cry range
            cry_inst_freqs = inst_freq[cry_bins, frame_idx]

            if len(cry_inst_freqs) > 10:
                # Measure variation (vibrato creates periodic variation)
                variation = cry_inst_freqs.std().item()

                # Baby cry FM typically has 5-15 Hz variation
                if 5 < variation < 20:
                    score = min(1.0, variation / 15)
                else:
                    score = 0.0
            else:
                score = 0.0

            fm_scores.append(score)

        return torch.tensor(fm_scores)

    def analyze_energy_distribution(self, audio: torch.Tensor,
                                   frame_length: int = 2048,
                                   hop_length: int = 512) -> torch.Tensor:
        """
        Analyze energy distribution to detect high concentration in 300-600 Hz.

        Args:
            audio: Input audio tensor
            frame_length: Frame size
            hop_length: Hop length

        Returns:
            Energy concentration scores per frame
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft) ** 2  # Power spectrum

        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]

        # Define frequency bands
        cry_band = (freqs >= 300) & (freqs <= 600)
        total_band = (freqs >= 100) & (freqs <= 4000)

        concentration_scores = []

        for frame_idx in range(magnitude.shape[1]):
            frame_power = magnitude[:, frame_idx]

            cry_energy = frame_power[cry_band].sum().item()
            total_energy = frame_power[total_band].sum().item()

            if total_energy > 1e-8:
                # Ratio of energy in cry band vs total
                concentration = cry_energy / total_energy
                # Baby cries should have >30% of energy in 300-600 Hz
                score = min(1.0, concentration / 0.4)
            else:
                score = 0.0

            concentration_scores.append(score)

        return torch.tensor(concentration_scores)

    def voice_activity_detection(self, audio: torch.Tensor,
                                frame_length: int = 1024,
                                threshold: float = 0.01) -> torch.Tensor:
        """
        Detect voice activity using energy and spectral features.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            threshold: Energy threshold for voice detection

        Returns:
            Binary mask indicating voice activity
        """
        # Frame the audio
        frames = audio.unfold(-1, frame_length, frame_length // 2)

        # Energy-based detection
        energy = torch.mean(frames ** 2, dim=-1)
        energy_thresh = torch.quantile(energy, 0.3) + threshold
        energy_mask = energy > energy_thresh

        # Spectral centroid for voice characteristics
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=frame_length // 2,
            return_complex=True
        )

        magnitude = torch.abs(stft)
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]

        # Compute spectral centroid
        spectral_centroid = torch.sum(magnitude * freqs.unsqueeze(-1), dim=0) / (torch.sum(magnitude, dim=0) + 1e-8)

        # Voice typically has centroid in 200-2000 Hz range
        centroid_mask = (spectral_centroid >= 200) & (spectral_centroid <= 2000)

        # Ensure masks have the same length by taking minimum
        min_length = min(len(energy_mask), len(centroid_mask))
        energy_mask = energy_mask[:min_length]
        centroid_mask = centroid_mask[:min_length]

        # Combine masks
        voice_mask = energy_mask & centroid_mask

        # Smooth the mask
        voice_mask_smooth = median_filter(voice_mask.numpy().astype(float), size=5) > 0.5

        return torch.from_numpy(voice_mask_smooth)

    def filter_adult_speech(self, audio: torch.Tensor,
                          frame_length: int = 2048,
                          hop_length: int = 512) -> torch.Tensor:
        """
        Detect and filter out adult speech based on lower F0 (80-250 Hz).
        Adult speech has fundamentally different pitch range than baby cries.

        Args:
            audio: Input audio tensor
            frame_length: Frame size
            hop_length: Hop length

        Returns:
            Confidence scores (0=adult speech, 1=not adult speech) per frame
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]

        # Define frequency bands
        adult_speech_band = (freqs >= 80) & (freqs <= 250)  # Adult F0 range
        baby_cry_band = (freqs >= 300) & (freqs <= 600)     # Baby cry F0 range

        rejection_scores = []

        for frame_idx in range(magnitude.shape[1]):
            frame_power = magnitude[:, frame_idx] ** 2

            adult_energy = frame_power[adult_speech_band].sum().item()
            baby_energy = frame_power[baby_cry_band].sum().item()
            total_energy = frame_power.sum().item()

            if total_energy > 1e-8:
                adult_ratio = adult_energy / total_energy
                baby_ratio = baby_energy / total_energy

                # If more energy in adult range than baby range, likely adult speech
                if adult_ratio > baby_ratio and adult_ratio > 0.2:
                    # Strong adult speech indicator
                    score = max(0.0, 1.0 - (adult_ratio / 0.4))
                else:
                    score = 1.0
            else:
                score = 1.0

            rejection_scores.append(score)

        return torch.tensor(rejection_scores)

    def filter_music(self, audio: torch.Tensor,
                   frame_length: int = 2048,
                   hop_length: int = 512,
                   stability_window: int = 20) -> torch.Tensor:
        """
        Detect and filter out music based on stable pitch patterns.
        Music has more stable pitch compared to cry's varying pitch contours.

        Args:
            audio: Input audio tensor
            frame_length: Frame size
            hop_length: Hop length
            stability_window: Number of frames to analyze for stability

        Returns:
            Confidence scores (0=music, 1=not music) per frame
        """
        # Track pitch over time using autocorrelation
        pitch_track = []

        for start_idx in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start_idx:start_idx + frame_length]

            # Autocorrelation for pitch detection
            autocorr = torch.nn.functional.conv1d(
                frame.unsqueeze(0).unsqueeze(0),
                frame.flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame_length - 1
            )[0, 0, frame_length-1:]

            # Search for pitch in wide range (100-2000 Hz)
            min_lag = int(self.sample_rate / 2000)
            max_lag = int(self.sample_rate / 100)

            if max_lag < len(autocorr):
                autocorr_range = autocorr[min_lag:max_lag]
                if autocorr_range.max() > 0:
                    peak_lag = min_lag + torch.argmax(autocorr_range).item()
                    pitch = self.sample_rate / peak_lag
                else:
                    pitch = 0.0
            else:
                pitch = 0.0

            pitch_track.append(pitch)

        pitch_track = torch.tensor(pitch_track)

        # Analyze pitch stability
        rejection_scores = []

        for i in range(len(pitch_track)):
            start_win = max(0, i - stability_window // 2)
            end_win = min(len(pitch_track), i + stability_window // 2)
            window = pitch_track[start_win:end_win]

            valid_pitches = window[window > 0]

            if len(valid_pitches) > stability_window // 2:
                # Compute pitch stability (coefficient of variation)
                pitch_mean = valid_pitches.mean().item()
                pitch_std = valid_pitches.std().item()

                if pitch_mean > 0:
                    cv = pitch_std / pitch_mean  # Coefficient of variation

                    # Music has very stable pitch (low CV < 0.05)
                    # Baby cries have varying pitch (CV > 0.1)
                    if cv < 0.05:
                        # Very stable = likely music
                        score = 0.2
                    elif cv < 0.1:
                        # Somewhat stable = possibly music
                        score = 0.5
                    else:
                        # Varying pitch = not music
                        score = 1.0
                else:
                    score = 1.0
            else:
                score = 1.0

            rejection_scores.append(score)

        return torch.tensor(rejection_scores)

    def filter_environmental_sounds(self, audio: torch.Tensor,
                                   frame_length: int = 2048,
                                   hop_length: int = 512) -> torch.Tensor:
        """
        Detect and filter out environmental sounds based on lack of harmonic structure.
        Environmental sounds (fan, traffic, white noise) lack clear harmonics.

        Args:
            audio: Input audio tensor
            frame_length: Frame size
            hop_length: Hop length

        Returns:
            Confidence scores (0=environmental, 1=not environmental) per frame
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]

        rejection_scores = []

        for frame_idx in range(magnitude.shape[1]):
            frame_mag = magnitude[:, frame_idx]

            # Measure spectral flatness (Wiener entropy)
            # High flatness = noise-like = environmental sound
            # Low flatness = tonal/harmonic = voice/cry
            geometric_mean = torch.exp(torch.mean(torch.log(frame_mag + 1e-10)))
            arithmetic_mean = torch.mean(frame_mag)

            if arithmetic_mean > 1e-8:
                spectral_flatness = geometric_mean / arithmetic_mean

                # Also check for harmonic structure
                # Look for peaks in spectrum (harmonics create peaks)
                # Smooth the spectrum and find peaks
                smoothed = torch.nn.functional.avg_pool1d(
                    frame_mag.unsqueeze(0).unsqueeze(0),
                    kernel_size=5,
                    stride=1,
                    padding=2
                ).squeeze()

                # Count significant peaks
                is_peak = (frame_mag[1:-1] > frame_mag[:-2]) & (frame_mag[1:-1] > frame_mag[2:])
                peak_heights = frame_mag[1:-1][is_peak]
                significant_peaks = (peak_heights > 0.3 * frame_mag.max()).sum().item()

                # Environmental sounds: high flatness (>0.5) and few peaks
                # Harmonic sounds (cries): low flatness (<0.3) and multiple peaks
                if spectral_flatness > 0.5 and significant_peaks < 3:
                    # Likely environmental noise
                    score = 0.3
                elif spectral_flatness < 0.3 and significant_peaks >= 4:
                    # Likely harmonic (cry or voice)
                    score = 1.0
                else:
                    # Ambiguous
                    score = 0.6
            else:
                score = 1.0

            rejection_scores.append(score)

        return torch.tensor(rejection_scores)

    def _merge_overlapping_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Merge overlapping time segments to avoid double-counting duration.

        Args:
            segments: List of (start_time, end_time) tuples

        Returns:
            List of non-overlapping merged (start_time, end_time) tuples
        """
        if not segments:
            return []

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x[0])

        merged = [sorted_segments[0]]

        for current_start, current_end in sorted_segments[1:]:
            last_start, last_end = merged[-1]

            # Check if current segment overlaps with the last merged segment
            if current_start <= last_end:
                # Merge by extending the end time if needed
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, add as new segment
                merged.append((current_start, current_end))

        return merged

    def classify_audio_segments(self, audio: torch.Tensor,
                               segment_duration: float = 3.0,
                               overlap: float = 0.5,
                               use_acoustic_validation: bool = True) -> List[Tuple[float, float, float, dict]]:
        """
        Classify audio segments using the trained model with acoustic validation.

        Args:
            audio: Input audio tensor
            segment_duration: Duration of each segment in seconds
            overlap: Overlap ratio between segments
            use_acoustic_validation: Whether to apply acoustic feature validation

        Returns:
            List of (start_time, end_time, cry_probability, metadata) tuples
            metadata contains: {'raw_prob', 'calibrated_prob', 'validated', 'rejection_reason'}
        """
        if self.model is None:
            print("Warning: No model loaded, skipping classification")
            return []

        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))

        segments = []

        for start_idx in range(0, len(audio) - segment_samples + 1, hop_samples):
            end_idx = start_idx + segment_samples
            segment = audio[start_idx:end_idx]

            # Preprocess segment
            try:
                mel_spec = self.preprocessor.extract_log_mel_spectrogram(segment)
                # Add batch dimension and channel dimension for model input (B, C, H, W)
                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)

                # Get prediction
                with torch.no_grad():
                    outputs = self.model(mel_spec)

                    # Apply calibration if available
                    if self.calibrator is not None:
                        probabilities = self.calibrator.calibrate_probabilities(outputs)
                        calibrated_prob = probabilities[0, 1].item() if self.config.NUM_CLASSES == 2 else probabilities[0, 0].item()
                    else:
                        probabilities = torch.softmax(outputs, dim=1)
                        calibrated_prob = probabilities[0, 1].item() if self.config.NUM_CLASSES == 2 else probabilities[0, 0].item()

                    # Store raw probability (before calibration)
                    raw_probabilities = torch.softmax(outputs, dim=1)
                    raw_prob = raw_probabilities[0, 1].item() if self.config.NUM_CLASSES == 2 else raw_probabilities[0, 0].item()

                start_time = start_idx / self.sample_rate
                end_time = end_idx / self.sample_rate

                # Extract acoustic features for validation
                metadata = {
                    'raw_prob': raw_prob,
                    'calibrated_prob': calibrated_prob,
                    'validated': True,
                    'rejection_reason': None
                }

                final_prob = calibrated_prob

                if use_acoustic_validation:
                    # Extract acoustic features for this segment
                    acoustic_features = self.acoustic_extractor.extract_all_features(segment)

                    # Apply acoustic validation
                    is_valid, adjusted_prob, reason = validate_cry_with_acoustic_features(
                        acoustic_features,
                        calibrated_prob,
                        threshold=0.5
                    )

                    metadata['validated'] = is_valid
                    metadata['rejection_reason'] = reason if not is_valid else None
                    metadata['acoustic_features'] = {
                        'pitch_mean': acoustic_features['pitch_mean'],
                        'hnr_mean': acoustic_features['hnr_mean'],
                        'duration': acoustic_features['duration']
                    }

                    final_prob = adjusted_prob

                segments.append((start_time, end_time, final_prob, metadata))

            except Exception as e:
                print(f"Error processing segment {start_idx}-{end_idx}: {e}")
                continue

        return segments

    def spectral_subtraction(self, audio: torch.Tensor,
                           noise_duration: float = 0.5) -> torch.Tensor:
        """
        Apply spectral subtraction to reduce background noise.

        Args:
            audio: Input audio tensor
            noise_duration: Duration of noise estimation period (seconds)

        Returns:
            Denoised audio tensor
        """
        # Estimate noise from beginning of audio
        noise_samples = int(noise_duration * self.sample_rate)
        noise_segment = audio[:noise_samples] if len(audio) > noise_samples else audio[:len(audio)//4]

        # Compute STFT of full audio and noise
        stft_audio = torch.stft(audio, n_fft=self.config.N_FFT,
                               hop_length=self.config.HOP_LENGTH, return_complex=True)
        stft_noise = torch.stft(noise_segment, n_fft=self.config.N_FFT,
                               hop_length=self.config.HOP_LENGTH, return_complex=True)

        # Estimate noise power spectrum
        noise_power = torch.mean(torch.abs(stft_noise) ** 2, dim=-1, keepdim=True)

        # Apply spectral subtraction
        audio_magnitude = torch.abs(stft_audio)
        audio_phase = torch.angle(stft_audio)

        # Subtract noise estimate (gentler for better audio quality)
        alpha = 1.0  # Reduced from 2.0 - less aggressive, better quality
        clean_magnitude = audio_magnitude - alpha * torch.sqrt(noise_power)

        # Apply higher spectral floor to preserve more audio content
        spectral_floor = 0.3 * audio_magnitude  # Increased from 0.1
        clean_magnitude = torch.maximum(clean_magnitude, spectral_floor)

        # Reconstruct complex spectrum
        clean_stft = clean_magnitude * torch.exp(1j * audio_phase)

        # Inverse STFT
        clean_audio = torch.istft(clean_stft, n_fft=self.config.N_FFT,
                                 hop_length=self.config.HOP_LENGTH, length=len(audio))

        return clean_audio

    def compute_acoustic_features(self, audio: torch.Tensor) -> dict:
        """
        Compute all acoustic features for baby cry detection.

        Args:
            audio: Input audio tensor

        Returns:
            Dictionary containing all acoustic feature scores
        """
        frame_length = 2048
        hop_length = 512

        print("  Computing harmonic structure...")
        harmonic_scores, f0_track = self.detect_harmonic_structure(audio, frame_length, hop_length)

        print("  Computing temporal patterns...")
        temporal_scores = self.detect_temporal_patterns(audio)

        print("  Computing pitch contours...")
        contour_scores, pitch_track = self.track_pitch_contours(audio, frame_length, hop_length)

        print("  Computing frequency modulation...")
        fm_scores = self.detect_frequency_modulation(audio, frame_length, hop_length)

        print("  Computing energy distribution...")
        energy_scores = self.analyze_energy_distribution(audio, frame_length, hop_length)

        print("  Computing rejection filters...")
        adult_rejection = self.filter_adult_speech(audio, frame_length, hop_length)
        music_rejection = self.filter_music(audio, frame_length, hop_length)
        env_rejection = self.filter_environmental_sounds(audio, frame_length, hop_length)

        return {
            'harmonic_scores': harmonic_scores,
            'f0_track': f0_track,
            'temporal_scores': temporal_scores,
            'contour_scores': contour_scores,
            'pitch_track': pitch_track,
            'fm_scores': fm_scores,
            'energy_scores': energy_scores,
            'adult_rejection': adult_rejection,
            'music_rejection': music_rejection,
            'env_rejection': env_rejection
        }

    def combine_acoustic_scores(self, features: dict, segment_duration: float = 3.0) -> List[Tuple[float, float, float]]:
        """
        Combine all acoustic features into final cry confidence scores.

        Args:
            features: Dictionary of acoustic features
            segment_duration: Duration of each segment in seconds

        Returns:
            List of (start_time, end_time, cry_confidence) tuples
        """
        # Get the shortest feature length for alignment
        min_len = min(
            len(features['harmonic_scores']),
            len(features['contour_scores']),
            len(features['fm_scores']),
            len(features['energy_scores']),
            len(features['adult_rejection']),
            len(features['music_rejection']),
            len(features['env_rejection'])
        )

        # Truncate all to same length
        harmonic = features['harmonic_scores'][:min_len]
        contour = features['contour_scores'][:min_len]
        fm = features['fm_scores'][:min_len]
        energy = features['energy_scores'][:min_len]
        adult_rej = features['adult_rejection'][:min_len]
        music_rej = features['music_rejection'][:min_len]
        env_rej = features['env_rejection'][:min_len]

        # Weighted combination of features
        # Baby cry indicators (positive weights)
        cry_indicators = (
            0.25 * harmonic +      # Harmonics are strong indicator
            0.15 * contour +       # Pitch contours
            0.10 * fm +            # Frequency modulation
            0.20 * energy          # Energy concentration in cry band
        )

        # Rejection filters (multiply by these to suppress non-cries)
        rejection_multiplier = adult_rej * music_rej * env_rej

        # Combined score
        combined_scores = cry_indicators * rejection_multiplier

        # Convert frame-level scores to segments
        hop_length = 512
        frame_duration = hop_length / self.sample_rate
        segment_frames = int(segment_duration / frame_duration)

        segments = []
        for i in range(0, len(combined_scores) - segment_frames, segment_frames // 2):
            segment_score = combined_scores[i:i+segment_frames].mean().item()
            start_time = i * frame_duration
            end_time = (i + segment_frames) * frame_duration

            segments.append((start_time, end_time, segment_score))

        return segments

    def isolate_baby_cry(self, audio: torch.Tensor,
                        cry_threshold: float = 0.7,
                        use_acoustic_features: bool = True) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """
        Main function to isolate baby cry from mixed audio.

        Args:
            audio: Input audio tensor
            cry_threshold: Probability threshold for cry detection
            use_acoustic_features: Whether to use acoustic feature-based filtering

        Returns:
            Tuple of (isolated_cry_audio, cry_time_segments)
        """
        print("Step 1: Applying spectral filtering...")
        # Step 1: Spectral filtering
        filtered_audio = self.spectral_filter(audio)

        print("Step 2: Voice activity detection...")
        # Step 2: Voice activity detection
        vad_mask = self.voice_activity_detection(filtered_audio)

        # Step 3: Acoustic feature analysis (NEW)
        acoustic_segments = []
        if use_acoustic_features:
            print("Step 3: Analyzing acoustic features...")
            features = self.compute_acoustic_features(filtered_audio)
            acoustic_segments = self.combine_acoustic_scores(features)
            print(f"  Found {len(acoustic_segments)} segments from acoustic analysis")

        print("Step 4: Classifying audio segments with ML model and acoustic validation...")
        # Step 4: Classify segments with ML model (now includes acoustic validation)
        ml_segments = self.classify_audio_segments(filtered_audio, use_acoustic_validation=use_acoustic_features)

        # Step 5: Process results
        print("Step 5: Processing results...")
        if use_acoustic_features and len(acoustic_segments) > 0:
            # Merge predictions from both methods
            combined_segments = []

            # Extract probabilities from ML segments (now includes metadata)
            ml_basic = [(s, e, p) for s, e, p, meta in ml_segments]

            for ac_start, ac_end, ac_score in acoustic_segments:
                # Find overlapping ML segment
                ml_score = 0.0
                for (ml_start, ml_end, ml_prob) in ml_basic:
                    # Check for overlap
                    if ml_start < ac_end and ml_end > ac_start:
                        ml_score = max(ml_score, ml_prob)

                # Weighted combination: 60% ML, 40% acoustic features
                # (ML model is more reliable if trained well)
                combined_score = 0.6 * ml_score + 0.4 * ac_score
                combined_segments.append((ac_start, ac_end, combined_score))

            segments = combined_segments
        else:
            # Use ML segments with acoustic validation already applied
            segments = [(s, e, p) for s, e, p, meta in ml_segments]

        print("Step 6: Applying spectral subtraction...")
        # Step 6: Noise reduction
        denoised_audio = self.spectral_subtraction(filtered_audio)

        print("Step 7: Extracting cry segments...")
        # Step 7: Extract cry segments
        print(f"Debug: Found {len(segments)} segments to evaluate:")
        for i, (start, end, prob) in enumerate(segments[:10]):  # Show first 10
            print(f"  Segment {i+1}: {start:.2f}s-{end:.2f}s, score={prob:.3f}, threshold={cry_threshold}")
        cry_segments = [(start, end) for start, end, prob in segments if prob > cry_threshold]
        print(f"Debug: After filtering with threshold {cry_threshold}: {len(cry_segments)} cry segments")

        # Create mask for cry regions
        cry_mask = torch.zeros_like(audio, dtype=torch.bool)
        for start_time, end_time in cry_segments:
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            cry_mask[start_idx:end_idx] = True

        # Apply masks
        isolated_audio = torch.zeros_like(denoised_audio)
        isolated_audio[cry_mask] = denoised_audio[cry_mask]

        return isolated_audio, cry_segments

    def isolate_baby_cry_multichannel(self, audio: torch.Tensor,
                                     cry_threshold: float = 0.7,
                                     use_acoustic_features: bool = True) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """
        Isolate baby cry from multi-channel audio while preserving phase relationships.

        This method processes the primary channel for cry detection but preserves
        all channels for sound localization with intact phase information.

        Args:
            audio: Multi-channel input audio tensor with shape (num_samples, num_channels)
            cry_threshold: Probability threshold for cry detection
            use_acoustic_features: Whether to use acoustic feature-based filtering

        Returns:
            Tuple of (isolated_multichannel_audio, cry_time_segments)
            All channels are preserved with phase relationships intact
        """
        # Handle both mono and multi-channel input
        if audio.dim() == 1:
            # Single channel - use original method
            return self.isolate_baby_cry(audio, cry_threshold, use_acoustic_features)

        # Multi-channel audio
        num_channels = audio.shape[1] if audio.dim() > 1 else 1

        # Process primary channel for cry detection
        primary_channel = audio[:, 0] if audio.dim() > 1 else audio

        print(f"Step 1: Applying spectral filtering (primary channel)...")
        # Spectral filtering on primary channel
        filtered_primary = self.spectral_filter(primary_channel)

        print("Step 2: Voice activity detection...")
        # VAD on primary channel
        vad_mask = self.voice_activity_detection(filtered_primary)

        # Acoustic feature analysis (NEW)
        acoustic_segments = []
        if use_acoustic_features:
            print("Step 3: Analyzing acoustic features (primary channel)...")
            features = self.compute_acoustic_features(filtered_primary)
            acoustic_segments = self.combine_acoustic_scores(features)
            print(f"  Found {len(acoustic_segments)} segments from acoustic analysis")

        print("Step 4: Classifying audio segments with ML model...")
        # Classify segments with ML model
        ml_segments = self.classify_audio_segments(filtered_primary, use_acoustic_validation=use_acoustic_features)

        # Process results
        print("Step 5: Processing results...")
        if use_acoustic_features and len(acoustic_segments) > 0:
            # Merge predictions from both methods
            ml_basic = [(s, e, p) for s, e, p, meta in ml_segments]

            combined_segments = []
            for ac_start, ac_end, ac_score in acoustic_segments:
                ml_score = 0.0
                for (ml_start, ml_end, ml_prob) in ml_basic:
                    if ml_start < ac_end and ml_end > ac_start:
                        ml_score = max(ml_score, ml_prob)

                combined_score = 0.6 * ml_score + 0.4 * ac_score
                combined_segments.append((ac_start, ac_end, combined_score))

            segments = combined_segments
        else:
            segments = [(s, e, p) for s, e, p, meta in ml_segments]

        print("Step 6: Applying spectral subtraction (primary channel)...")
        # Noise reduction on primary channel
        denoised_primary = self.spectral_subtraction(filtered_primary)

        print("Step 7: Extracting cry segments...")
        # Extract cry segments
        print(f"Debug: Found {len(segments)} segments to evaluate:")
        for i, (start, end, prob) in enumerate(segments[:10]):
            print(f"  Segment {i+1}: {start:.2f}s-{end:.2f}s, score={prob:.3f}, threshold={cry_threshold}")

        cry_segments = [(start, end) for start, end, prob in segments if prob > cry_threshold]
        print(f"Debug: After filtering: {len(cry_segments)} cry segments")

        # Create mask for cry regions
        cry_mask = torch.zeros(len(denoised_primary), dtype=torch.bool)
        for start_time, end_time in cry_segments:
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            cry_mask[start_idx:end_idx] = True

        # Apply mask to ALL channels (preserving phase)
        print(f"Step 8: Applying cry mask to {num_channels} channels...")
        isolated_audio_multichannel = torch.zeros_like(audio)

        for ch in range(num_channels):
            # For each channel, apply the same cry mask
            isolated_audio_multichannel[cry_mask, ch] = audio[cry_mask, ch]

        return isolated_audio_multichannel, cry_segments

    def process_audio_file(self, input_path: str, output_path: str,
                          cry_threshold: float = 0.7,
                          use_acoustic_features: bool = True) -> dict:
        """
        Process an audio file to extract baby cries.

        Args:
            input_path: Path to input audio file
            output_path: Path to save filtered audio
            cry_threshold: Probability threshold for cry detection
            use_acoustic_features: Whether to use acoustic feature-based filtering

        Returns:
            Processing results dictionary
        """
        print(f"Processing audio file: {input_path}")
        print(f"Acoustic features: {'ENABLED' if use_acoustic_features else 'DISABLED'}")

        # Load audio
        audio, sample_rate = torchaudio.load(input_path)
        # Preserve multi-channel if present, otherwise use first channel
        if audio.shape[0] > 1:
            # Multi-channel - preserve all channels
            audio = audio.transpose(0, 1)  # (channels, samples) -> (samples, channels)
            print(f"Loaded multi-channel audio: {audio.shape}")
        else:
            # Single channel
            audio = audio[0]
            print(f"Loaded mono audio: {audio.shape}")

        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            if audio.dim() > 1:
                # Multi-channel resample
                resampled = []
                for ch in range(audio.shape[1]):
                    resampled.append(resampler(audio[:, ch]))
                audio = torch.stack(resampled, dim=1)
            else:
                audio = resampler(audio)

        # Process audio with acoustic features
        if audio.dim() > 1:
            # Multi-channel
            isolated_audio, cry_segments = self.isolate_baby_cry_multichannel(
                audio,
                cry_threshold,
                use_acoustic_features=use_acoustic_features
            )
        else:
            # Mono
            isolated_audio, cry_segments = self.isolate_baby_cry(
                audio,
                cry_threshold,
                use_acoustic_features=use_acoustic_features
            )

        # Save results
        if isolated_audio.dim() > 1:
            # Multi-channel save
            torchaudio.save(output_path, isolated_audio.transpose(0, 1), self.sample_rate)
        else:
            # Mono save
            torchaudio.save(output_path, isolated_audio.unsqueeze(0), self.sample_rate)

        # Calculate statistics
        total_duration = len(audio) / self.sample_rate

        # Merge overlapping cry segments before calculating duration
        # (segments overlap by 50% due to sliding window approach)
        merged_segments = self._merge_overlapping_segments(cry_segments)
        cry_duration = sum(end - start for start, end in merged_segments)
        cry_percentage = (cry_duration / total_duration) * 100 if total_duration > 0 else 0

        results = {
            'input_file': input_path,
            'output_file': output_path,
            'total_duration': total_duration,
            'cry_duration': cry_duration,
            'cry_percentage': cry_percentage,
            'cry_segments': cry_segments,
            'num_cry_segments': len(cry_segments),
            'acoustic_features_used': use_acoustic_features
        }

        print(f"Processing complete:")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Cry duration: {cry_duration:.2f}s ({cry_percentage:.1f}%)")
        print(f"  Cry segments found: {len(cry_segments)}")

        return results


def create_audio_filter(config: Config = Config(), model_path: Optional[str] = None,
                       calibrator_path: Optional[str] = None) -> BabyCryAudioFilter:
    """
    Create and return a baby cry audio filter.

    Args:
        config: Configuration object
        model_path: Path to trained model
        calibrator_path: Path to confidence calibrator

    Returns:
        Initialized audio filter
    """
    return BabyCryAudioFilter(config, model_path, calibrator_path)


if __name__ == "__main__":
    # Example usage
    config = Config()

    # Initialize filter (replace with your model path)
    model_path = "results/latest/model_best.pth"
    audio_filter = create_audio_filter(config, model_path)

    # Process an audio file
    input_file = "test_audio.wav"
    output_file = "filtered_cry.wav"

    if os.path.exists(input_file):
        results = audio_filter.process_audio_file(input_file, output_file)
        print(f"Results: {results}")
    else:
        print(f"Test file {input_file} not found")