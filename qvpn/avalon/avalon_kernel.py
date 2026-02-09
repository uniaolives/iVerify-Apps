# avalon_kernel.py

import numpy as np
import json
import os
import time
import pickle
from datetime import datetime
from .components.eeg_processor import EEGProcessor
from .components.ricci_flow import RicciFlowEngine
from .components.neural_field import QuantumNeuralField
from .components.audio_synth import HarmonicSynthesizer
from .components.visualizer import ManifoldVisualizer

class AvalonKernel:
    """Main engine of Project Avalon"""

    def __init__(self, config_file="avalon_config/default_config.json"):
        self.config = self.load_config(config_file)
        self.state = {
            'phase': 'initializing',
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'manifold_state': None,
            'user_state': {},
            'therapy_progress': 0.0
        }

        self.initialize_components()

    def load_config(self, config_file):
        """Load configuration from file"""
        default_config = {
            "system": {
                "sampling_rate": 1000,
                "buffer_size": 1024,
                "update_rate": 20,
                "max_sessions": 100
            },
            "therapy": {
                "session_duration": 60,  # 1 minute for test
                "ricci_flow_rate": 0.1,
                "curvature_threshold": 0.05,
                "healing_protocol": "adaptive"
            },
            "audio": {
                "base_frequency": 110,
                "harmonic_series": 8,
                "binaural_beats": True
            },
            "visual": {
                "resolution": [1920, 1080],
                "color_scheme": "aurora"
            }
        }

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        return default_config

    def initialize_components(self):
        """Initialize all system components"""
        print("🔄 Initializing Avalon components...")
        self.eeg_processor = EEGProcessor(
            sampling_rate=self.config['system']['sampling_rate'],
            buffer_size=self.config['system']['buffer_size']
        )
        self.ricci_flow = RicciFlowEngine(
            flow_rate=self.config['therapy']['ricci_flow_rate']
        )
        self.neural_field = QuantumNeuralField()
        self.audio_synthesizer = HarmonicSynthesizer(
            base_freq=self.config['audio']['base_frequency'],
            harmonics=self.config['audio']['harmonic_series']
        )
        self.visualizer = ManifoldVisualizer(
            resolution=self.config['visual']['resolution'],
            color_scheme=self.config['visual']['color_scheme']
        )
        print("✅ All components initialized!")

    def initialize_manifold(self):
        """Initialize the neural manifold state"""
        N = 50
        x = np.linspace(-5, 5, N)
        y = np.linspace(-5, 5, N)
        X, Y = np.meshgrid(x, y)
        Z = 0.5 * (X**2 + Y**2)
        Z -= 2 * np.exp(-(X**2 + Y**2) / 2) # depression well
        return Z

    def start_session(self, user_profile=None):
        print(f"\n🧠 STARTING AVALON SESSION {self.state['session_id']}")
        self.state['phase'] = 'active'
        self.state['manifold_state'] = self.initialize_manifold()
        self.run_therapy_loop()

    def run_therapy_loop(self):
        session_duration = self.config['therapy']['session_duration']
        update_interval = 1.0 / self.config['system']['update_rate']
        start_time = time.time()

        try:
            while self.state['phase'] == 'active':
                elapsed = time.time() - start_time
                self.state['therapy_progress'] = min(1.0, elapsed / session_duration)

                eeg_data = self.eeg_processor.get_latest_data()

                # Mock metrics
                metrics = {
                    'stress_level': max(0, min(1, 0.7 * (1 - self.state['therapy_progress']))),
                    'focus_level': 0.3 + 0.6 * self.state['therapy_progress'],
                    'coherence_alpha': 0.4 + 0.3 * self.state['therapy_progress']
                }

                self.state['manifold_state'] = self.ricci_flow.apply(
                    self.state['manifold_state'],
                    rate=self.config['therapy']['ricci_flow_rate'],
                    focus=metrics['focus_level']
                )

                if elapsed >= session_duration:
                    print("\n✅ Session complete!")
                    self.end_session()
                    break

                if int(elapsed) % 5 == 0:
                    print(f"\r⏱️  {elapsed:.0f}s | Progress: {self.state['therapy_progress']*100:.1f}% | Stress: {metrics['stress_level']:.2f}", end='')

                time.sleep(update_interval)
        except KeyboardInterrupt:
            self.end_session()

    def end_session(self):
        print(f"\nSession {self.state['session_id']} ended.")
        self.save_session_data()
        self.state['phase'] = 'completed'

    def save_session_data(self):
        os.makedirs("qvpn/avalon/session_data", exist_ok=True)
        filename = f"qvpn/avalon/session_data/session_{self.state['session_id']}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump({'state': self.state}, f)
        print(f"💾 Session data saved to {filename}")
