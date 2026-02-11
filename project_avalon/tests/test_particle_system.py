import pytest
import numpy as np
from project_avalon.core.unified_particle_system import UnifiedParticleSystem
from project_avalon.gui.view_3d import ConsciousnessVisualizer3D
from dataclasses import dataclass

@dataclass
class MockEEG:
    attention: float
    meditation: float

def test_particle_system_initialization():
    ps = UnifiedParticleSystem(num_particles=120)
    assert len(ps.particles) == 120
    assert ps.current_mode == "MANDALA"
    assert ps.transition_progress == 1.0

def test_particle_system_transitions():
    ps = UnifiedParticleSystem(num_particles=10)

    # Inicia transição para DNA
    ps.set_mode("DNA")
    assert ps.target_mode == "DNA"
    assert ps.transition_progress == 0.0

    # Simula frames de transição
    ps.update(0.1)
    assert ps.transition_progress > 0.0
    assert ps.current_mode == "MANDALA" # Ainda não terminou

    # Força término da transição
    for _ in range(100):
        ps.update(0.1)
        if ps.transition_progress >= 1.0:
            break

    assert ps.current_mode == "DNA"
    assert ps.transition_progress == 1.0

def test_geometry_validity():
    ps = UnifiedParticleSystem(num_particles=120)

    modes = ["MANDALA", "DNA", "HYPERCORE"]
    for mode in modes:
        ps.set_mode(mode)
        # Força estabilidade do modo
        ps.transition_progress = 1.0
        ps.current_mode = mode

        ps.update(0.1)
        data = ps.get_particle_data()

        positions = np.array(data['positions'])
        assert positions.shape == (120, 3)
        # Verifica se não há NaNs
        assert not np.isnan(positions).any()
        # Verifica se há movimento (não todos em zero)
        assert np.linalg.norm(positions) > 0.001

def test_visualizer_integration():
    viz = ConsciousnessVisualizer3D(num_particles=50)

    # Simula foco (DNA)
    eeg_focus = MockEEG(attention=85, meditation=20)
    viz.update_from_eeg(eeg_focus)
    assert viz.particle_system.target_mode == "DNA"

    # Simula meditação (HYPERCORE)
    eeg_zen = MockEEG(attention=10, meditation=90)
    viz.update_from_eeg(eeg_zen)
    assert viz.particle_system.target_mode == "HYPERCORE"

    # Simula repouso (MANDALA)
    eeg_rest = MockEEG(attention=15, meditation=15)
    viz.update_from_eeg(eeg_rest)
    assert viz.particle_system.target_mode == "MANDALA"

    # Render frame
    frame_data = viz.render_frame(0.016) # ~60fps
    assert len(frame_data['positions']) == 50
    assert frame_data['mode'] in ["MANDALA", "DNA", "HYPERCORE"]
