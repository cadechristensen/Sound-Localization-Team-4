"""
Integrated Robot Baby Monitor System
Combines real-time baby cry detection with sound localization for robot navigation.
Optimized for Raspberry Pi 5 with TI PCM6260-Q1 microphone array.
"""

import multiprocessing as mp
import logging
import time
import signal
import sys
import argparse
from pathlib import Path
from typing import Optional

from realtime_baby_cry_detector import RealtimeBabyCryDetector, DetectionResult
from sound_localization_interface import run_localization_process, LocalizationResult


class RobotBabyMonitor:
    """
    Complete baby monitoring system integrating cry detection and sound localization.
    """

    def __init__(
        self,
        model_path: str,
        audio_device_index: Optional[int] = None,
        num_channels: int = 4,
        detection_threshold: float = 0.75,
        confirmation_threshold: float = 0.85,
        device: str = 'cpu'
    ):
        """
        Initialize robot baby monitor.

        Args:
            model_path: Path to trained cry detection model
            audio_device_index: Microphone array device index
            num_channels: Number of microphone channels
            detection_threshold: Initial detection threshold
            confirmation_threshold: Confirmation threshold for wake-up
            device: Processing device (cpu/cuda)
        """
        self.model_path = model_path
        self.audio_device_index = audio_device_index
        self.num_channels = num_channels
        self.detection_threshold = detection_threshold
        self.confirmation_threshold = confirmation_threshold
        self.device = device

        # IPC queue for detection -> localization
        self.detection_queue = mp.Queue(maxsize=10)

        # Process handles
        self.localization_process = None
        self.detector = None

        logging.info("Robot Baby Monitor initialized")

    def start_localization_process(self):
        """Start sound localization in separate process."""
        logging.info("Starting sound localization process...")

        self.localization_process = mp.Process(
            target=run_localization_process,
            args=(self.detection_queue,),
            daemon=True
        )
        self.localization_process.start()

        logging.info("Sound localization process started")

    def start_cry_detector(self):
        """Start real-time cry detector."""
        logging.info("Starting baby cry detector...")

        self.detector = RealtimeBabyCryDetector(
            model_path=self.model_path,
            use_tta=True,  # Use TTA for confirmation
            detection_threshold=self.detection_threshold,
            confirmation_threshold=self.confirmation_threshold,
            device=self.device,
            audio_device_index=self.audio_device_index,
            num_channels=self.num_channels
        )

        # Set detection queue for localization
        self.detector.detection_queue = self.detection_queue

        # Optional: Add custom callback
        def on_detection(detection: DetectionResult):
            print(f"\n{'='*70}")
            print(f"BABY CRY DETECTED!")
            print(f"{'='*70}")
            print(f"Confidence: {detection.confidence:.1%}")
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print(f"Status: Waking robot and running localization...")
            print(f"{'='*70}\n")

        self.detector.on_cry_detected = on_detection

        # Start detector
        self.detector.start(stream_audio=True)

        logging.info("Baby cry detector started")

    def start(self):
        """Start the complete monitoring system."""
        print("\n" + "="*70)
        print("ROBOT BABY MONITOR - STARTING")
        print("="*70)
        print(f"Model: {Path(self.model_path).name}")
        print(f"Microphone Channels: {self.num_channels}")
        print(f"Detection Threshold: {self.detection_threshold:.0%}")
        print(f"Confirmation Threshold: {self.confirmation_threshold:.0%}")
        print(f"Processing Device: {self.device}")
        print("="*70 + "\n")

        # Start localization process
        self.start_localization_process()

        # Small delay to ensure process is ready
        time.sleep(1)

        # Start cry detector
        self.start_cry_detector()

        print("\n" + "="*70)
        print("SYSTEM ACTIVE - LOW POWER LISTENING MODE")
        print("="*70)
        print("Listening for baby cries...")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")

    def stop(self):
        """Stop the monitoring system."""
        logging.info("Stopping Robot Baby Monitor...")

        # Stop detector
        if self.detector:
            self.detector.stop()

        # Terminate localization process
        if self.localization_process and self.localization_process.is_alive():
            self.localization_process.terminate()
            self.localization_process.join(timeout=2.0)

        logging.info("Robot Baby Monitor stopped")

    def run(self):
        """Run the monitoring system."""
        try:
            self.start()

            # Keep running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("SHUTDOWN INITIATED")
            print("="*70)
            self.stop()
            print("\nSystem stopped successfully\n")


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    print("\n\nReceived shutdown signal...")
    sys.exit(0)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Robot Baby Monitor - Integrated Cry Detection & Localization'
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained baby cry detection model')
    parser.add_argument('--device-index', type=int, default=None,
                       help='Audio device index for microphone array')
    parser.add_argument('--channels', type=int, default=4,
                       help='Number of microphone channels (default: 4)')
    parser.add_argument('--detection-threshold', type=float, default=0.75,
                       help='Detection threshold (default: 0.75)')
    parser.add_argument('--confirmation-threshold', type=float, default=0.85,
                       help='Confirmation threshold for wake-up (default: 0.85)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Processing device (default: cpu)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Verify model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Create and run monitor
    monitor = RobotBabyMonitor(
        model_path=args.model,
        audio_device_index=args.device_index,
        num_channels=args.channels,
        detection_threshold=args.detection_threshold,
        confirmation_threshold=args.confirmation_threshold,
        device=args.device
    )

    monitor.run()


if __name__ == "__main__":
    main()
