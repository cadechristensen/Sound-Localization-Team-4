"""
Test your own audio files with acoustic feature-based filtering.
"""

# Add project root to path
import sys
import os
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import sys
from pathlib import Path
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from src.config import Config
from src.audio_filter import BabyCryAudioFilter
from src.data_preprocessing import AudioPreprocessor


def _merge_overlapping_segments(segments):
    """Merge overlapping time segments to avoid double-counting duration."""
    if not segments:
        return []
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]
    for current_start, current_end in sorted_segments[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged


def visualize_filtering_pipeline(input_path: str, audio_filter: BabyCryAudioFilter,
                                 cry_threshold: float = 0.5,
                                 use_acoustic_features: bool = True,
                                 processing_results: dict = None,
                                 show_ml_predictions: bool = False):
    """
    Generate visualization plots showing all filtering stages.

    Args:
        input_path: Path to audio file
        audio_filter: Initialized BabyCryAudioFilter
        cry_threshold: Detection threshold
        use_acoustic_features: Whether acoustic features were used
        processing_results: Results from process_audio_file (optional, for consistency)
        show_ml_predictions: Whether to show the ML predictions plot (default: False)
    """
    print("\nGenerating visualization plots...")

    # Load and prepare audio
    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] > 1:
        audio = audio[0]  # Use first channel for visualization
    else:
        audio = audio[0]

    # Resample if needed
    if sr != audio_filter.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, audio_filter.sample_rate)
        audio = resampler(audio)
        sr = audio_filter.sample_rate

    # Step through the filtering pipeline
    print("  Step 1: Spectral filtering...")
    filtered_audio = audio_filter.spectral_filter(audio)

    print("  Step 2: Voice activity detection...")
    vad_mask = audio_filter.voice_activity_detection(filtered_audio)

    print("  Step 3: Spectral subtraction...")
    denoised_audio = audio_filter.spectral_subtraction(filtered_audio)

    # Get acoustic features if enabled
    acoustic_features = None
    if use_acoustic_features:
        print("  Step 4: Computing acoustic features...")
        acoustic_features = audio_filter.compute_acoustic_features(filtered_audio)

    # Get ML predictions
    print("  Step 5: Getting ML predictions...")
    ml_segments = audio_filter.classify_audio_segments(
        filtered_audio,
        use_acoustic_validation=use_acoustic_features
    )

    # Create visualization with dynamic numbering
    plt.figure(figsize=(18, 12))

    # Track plot number dynamically
    plot_num = 0
    preprocessor = AudioPreprocessor(audio_filter.config)
    time_axis = np.arange(len(audio)) / sr

    # Plot: Original waveform
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    ax.plot(time_axis, audio.numpy(), linewidth=0.5, color='blue', alpha=0.7)
    ax.set_title(f'{plot_num}. Original Waveform', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.grid(alpha=0.3)

    # Plot: Spectral filtered waveform
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    time_axis_filtered = np.arange(len(filtered_audio)) / sr
    ax.plot(time_axis_filtered, filtered_audio.numpy(), linewidth=0.5, color='green', alpha=0.7)
    ax.set_title(f'{plot_num}. Spectral Filtered (100-3000 Hz)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.grid(alpha=0.3)

    # Plot: Denoised waveform
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    time_axis_denoised = np.arange(len(denoised_audio)) / sr
    ax.plot(time_axis_denoised, denoised_audio.numpy(), linewidth=0.5, color='purple', alpha=0.7)
    ax.set_title(f'{plot_num}. Spectral Subtraction Applied', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.grid(alpha=0.3)

    # Plot: Original spectrogram
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    original_spec = preprocessor.extract_log_mel_spectrogram(audio)
    im = ax.imshow(original_spec.numpy(), aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, len(audio) / sr, 0, original_spec.shape[0]])
    ax.set_title(f'{plot_num}. Original Mel Spectrogram', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Mel Frequency Bin', fontsize=9)
    plt.colorbar(im, ax=ax, label='dB')

    # Plot: Filtered spectrogram
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    filtered_spec = preprocessor.extract_log_mel_spectrogram(filtered_audio)
    im = ax.imshow(filtered_spec.numpy(), aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, len(filtered_audio) / sr, 0, filtered_spec.shape[0]])
    ax.set_title(f'{plot_num}. Filtered Mel Spectrogram', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Mel Frequency Bin', fontsize=9)
    plt.colorbar(im, ax=ax, label='dB')

    # Plot: Denoised spectrogram
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    denoised_spec = preprocessor.extract_log_mel_spectrogram(denoised_audio)
    im = ax.imshow(denoised_spec.numpy(), aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, len(denoised_audio) / sr, 0, denoised_spec.shape[0]])
    ax.set_title(f'{plot_num}. Denoised Mel Spectrogram', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Mel Frequency Bin', fontsize=9)
    plt.colorbar(im, ax=ax, label='dB')

    # Plot: Voice activity detection
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    # Create time axis matching VAD mask length
    frame_length = 1024
    hop_length_vad = frame_length // 2
    vad_time = np.arange(len(vad_mask)) * hop_length_vad / sr

    # Ensure arrays have matching lengths
    if len(vad_time) != len(vad_mask):
        min_len = min(len(vad_time), len(vad_mask))
        vad_time = vad_time[:min_len]
        vad_mask_plot = vad_mask[:min_len]
    else:
        vad_mask_plot = vad_mask

    # Plot waveform in background (normalized to 0-1 range)
    audio_normalized = (audio.numpy() - audio.numpy().min()) / (audio.numpy().max() - audio.numpy().min() + 1e-8)
    ax.plot(time_axis, audio_normalized, linewidth=0.5, color='gray', alpha=0.4, label='Waveform')

    # Overlay VAD mask
    ax.fill_between(vad_time, 0, 1, where=vad_mask_plot.numpy(), alpha=0.3, color='green', label='Voice Activity')
    ax.set_title(f'{plot_num}. Voice Activity Detection', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Activity', fontsize=9)
    ax.set_xlim([0, len(audio) / sr])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot: ML predictions (optional - only if enabled)
    if show_ml_predictions:
        plot_num += 1
        ax = plt.subplot(4, 3, plot_num)
        for start, end, prob, _meta in ml_segments:
            color = 'red' if prob > cry_threshold else 'orange'
            alpha = min(0.8, prob)
            ax.axvspan(start, end, alpha=alpha, color=color)
        ax.set_title(f'{plot_num}. ML Predictions (threshold={cry_threshold})', fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Cry Probability', fontsize=9)
        ax.set_xlim([0, len(audio) / sr])
        ax.set_ylim([0, 1])
        ax.axhline(y=cry_threshold, color='black', linestyle='--', label='Threshold', linewidth=1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Plot: Acoustic features (if enabled)
    if acoustic_features:
        plot_num += 1
        ax = plt.subplot(4, 3, plot_num)

        # Show harmonic scores
        harmonic = acoustic_features['harmonic_scores']
        energy = acoustic_features['energy_scores']
        hop_length = 512

        # Use the shorter length for alignment
        min_len_features = min(len(harmonic), len(energy))
        time_axis_features = np.arange(min_len_features) * hop_length / sr

        ax.plot(time_axis_features, harmonic[:min_len_features].numpy(), label='Harmonic', alpha=0.7)
        ax.plot(time_axis_features, energy[:min_len_features].numpy(),
                label='Energy', alpha=0.7)
        ax.set_title(f'{plot_num}. Acoustic Features', fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Score', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Plot: Rejection filters (if enabled)
    if acoustic_features:
        plot_num += 1
        ax = plt.subplot(4, 3, plot_num)

        # Get all rejection filters
        adult_rej = acoustic_features['adult_rejection']
        music_rej = acoustic_features['music_rejection']
        env_rej = acoustic_features['env_rejection']

        # Use the shortest length for alignment
        min_len_rejection = min(len(adult_rej), len(music_rej), len(env_rej))
        hop_length = 512
        time_axis_rejection = np.arange(min_len_rejection) * hop_length / sr

        ax.plot(time_axis_rejection, adult_rej[:min_len_rejection].numpy(),
                 label='Adult Speech Filter', alpha=0.7)
        ax.plot(time_axis_rejection, music_rej[:min_len_rejection].numpy(),
                 label='Music Filter', alpha=0.7)
        ax.plot(time_axis_rejection, env_rej[:min_len_rejection].numpy(),
                 label='Environmental Filter', alpha=0.7)
        ax.set_title(f'{plot_num}. Rejection Filters (1=keep, 0=reject)', fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Score', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Plot: Final cry segments
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    cry_segments = [(start, end) for start, end, prob, _meta in ml_segments if prob > cry_threshold]

    # Show waveform with cry regions highlighted
    ax.plot(time_axis, audio.numpy(), linewidth=0.5, color='gray', alpha=0.5, label='Original')
    for start, end in cry_segments:
        ax.axvspan(start, end, alpha=0.3, color='red')

    ax.set_title(f'{plot_num}. Detected Cry Segments ({len(cry_segments)} found)',
                  fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.grid(alpha=0.3)

    # Plot: Summary statistics
    plot_num += 1
    ax = plt.subplot(4, 3, plot_num)
    ax.axis('off')

    # Use processing_results if provided, otherwise compute from visualization run
    if processing_results:
        total_duration = processing_results['total_duration']
        cry_duration = processing_results['cry_duration']
        cry_percentage = processing_results['cry_percentage']
        num_segments = processing_results['num_cry_segments']
    else:
        total_duration = len(audio) / sr
        # Merge overlapping segments before calculating duration
        merged_segments = _merge_overlapping_segments(cry_segments)
        cry_duration = sum(end - start for start, end in merged_segments)
        cry_percentage = (cry_duration / total_duration) * 100 if total_duration > 0 else 0
        num_segments = len(cry_segments)

    summary_text = f"""
    FILTERING SUMMARY
    ═════════════════════════════

    Total Duration: {total_duration:.2f}s
    Cry Duration: {cry_duration:.2f}s
    Cry Percentage: {cry_percentage:.1f}%

    Cry Segments: {num_segments}
    Threshold: {cry_threshold}

    Filters Applied:
    ✓ Spectral filtering (100-3000 Hz)
    ✓ Voice activity detection
    ✓ Spectral subtraction
    ✓ ML model classification
    {'✓ Acoustic feature validation' if use_acoustic_features else '✗ Acoustic features (disabled)'}
    {'✓ Adult speech rejection' if use_acoustic_features else '✗ Adult speech rejection'}
    {'✓ Music rejection' if use_acoustic_features else '✗ Music rejection'}
    {'✓ Environmental rejection' if use_acoustic_features else '✗ Environmental rejection'}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center')
    ax.set_title(f'{plot_num}. Summary', fontweight='bold', fontsize=10)

    plt.tight_layout()

    # Save figure
    input_file = Path(input_path)
    output_dir = input_file.parent / 'filtering_visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{input_file.stem}_filtering_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.close()

    return output_path


def test_audio_file(input_path: str, model_path: str = None,
                   cry_threshold: float = 0.5,
                   use_acoustic_features: bool = True,
                   generate_plots: bool = False):
    """
    Test a single audio file for baby cry detection.

    Args:
        input_path: Path to your audio file
        model_path: Path to trained model (optional - can work without ML model)
        cry_threshold: Detection threshold (0.0-1.0, lower = more sensitive)
        use_acoustic_features: Enable acoustic feature analysis
        generate_plots: Generate visualization plots of filtering stages
    """
    # Check if input file exists
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        return

    # Initialize
    config = Config()
    config.USE_ACOUSTIC_FEATURES = use_acoustic_features

    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path)

    # Generate output filename
    input_file = Path(input_path)
    output_path = input_file.parent / f"{input_file.stem}_filtered{input_file.suffix}"

    print("=" * 80)
    print(f"Testing Audio File: {input_path}")
    print("=" * 80)
    print(f"Model: {'Using ML model' if model_path else 'Acoustic features only'}")
    print(f"Acoustic features: {'ENABLED' if use_acoustic_features else 'DISABLED'}")
    print(f"Threshold: {cry_threshold}")
    print(f"Output will be saved to: {output_path}")
    print()

    # Process the audio
    try:
        results = audio_filter.process_audio_file(
            input_path=str(input_path),
            output_path=str(output_path),
            cry_threshold=cry_threshold,
            use_acoustic_features=use_acoustic_features
        )

        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Total duration: {results['total_duration']:.2f} seconds")
        print(f"Cry duration: {results['cry_duration']:.2f} seconds")
        print(f"Cry percentage: {results['cry_percentage']:.1f}%")
        print(f"Number of cry segments: {results['num_cry_segments']}")

        if results['cry_segments']:
            print("\nCry segments detected:")
            for i, (start, end) in enumerate(results['cry_segments'][:10], 1):
                duration = end - start
                print(f"  {i}. {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")

            if len(results['cry_segments']) > 10:
                print(f"  ... and {len(results['cry_segments']) - 10} more segments")

        print(f"\nFiltered audio saved to: {output_path}")

        # Generate visualization plots if requested
        if generate_plots:
            try:
                plot_path = visualize_filtering_pipeline(
                    input_path,
                    audio_filter,
                    cry_threshold,
                    use_acoustic_features,
                    results  # Pass the actual results
                )
                print(f"Visualization plots saved to: {plot_path}")
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")
                import traceback
                traceback.print_exc()

        return results

    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_files(audio_dir: str, model_path: str = None):
    """
    Test all audio files in a directory.

    Args:
        audio_dir: Directory containing audio files
        model_path: Path to trained model (optional)
    """
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        print(f"Error: Directory not found: {audio_dir}")
        return

    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print()

    # Process each file
    all_results = []
    for audio_file in audio_files:
        results = test_audio_file(str(audio_file), model_path)
        if results:
            all_results.append(results)
        print("\n" + "-" * 80 + "\n")

    # Summary
    if all_results:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        total_files = len(all_results)
        files_with_cries = sum(1 for r in all_results if r['num_cry_segments'] > 0)
        total_segments = sum(r['num_cry_segments'] for r in all_results)

        print(f"Files processed: {total_files}")
        print(f"Files with detected cries: {files_with_cries}")
        print(f"Total cry segments: {total_segments}")


if __name__ == "__main__":
    # Example usage - modify these paths for your setup

    # Option 1: Test a single file
    print("OPTION 1: Test a single audio file")
    print("-" * 80)

    # Replace with your audio file path
    INPUT_AUDIO = "path/to/your/audio.wav"

    # Replace with your trained model path (or set to None for acoustic-only)
    MODEL_PATH = "results/model_best.pth"  # or None

    # Adjust threshold (lower = more sensitive, higher = fewer false positives)
    THRESHOLD = 0.5

    # Test single file
    if len(sys.argv) > 1:
        # Use command line argument if provided
        input_file = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != 'None' else None
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        generate_plots = '--plot' in sys.argv or '-p' in sys.argv

        test_audio_file(input_file, model_path, threshold, generate_plots=generate_plots)
    else:
        # Use example path
        print("\nUsage:")
        print("  python test_my_audio.py <audio_file> [model_path] [threshold] [--plot]")
        print("\nOptions:")
        print("  --plot, -p    Generate visualization plots of filtering stages")
        print("\nExamples:")
        print("  python test_my_audio.py my_baby_cry.wav")
        print("  python test_my_audio.py my_baby_cry.wav results/model_best.pth")
        print("  python test_my_audio.py my_baby_cry.wav results/model_best.pth 0.7")
        print("  python test_my_audio.py my_baby_cry.wav None 0.5  # Acoustic only")
        print("  python test_my_audio.py my_baby_cry.wav results/model_best.pth 0.5 --plot  # With plots")
        print()

        # If you want to test without command line args, uncomment and modify:
        # test_audio_file(INPUT_AUDIO, MODEL_PATH, THRESHOLD)

    # Option 2: Test all files in a directory (uncomment to use)
    # print("\n\nOPTION 2: Test all audio files in a directory")
    # print("-" * 80)
    # test_multiple_files("path/to/audio/directory", MODEL_PATH)
