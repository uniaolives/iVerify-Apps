import asyncio
import numpy as np
import pytest
from project_avalon.components.neural_emotion_engine import NeuralQuantumAnalyzer

async def test_neural_oracle_integration():
    analyzer = NeuralQuantumAnalyzer(user_id="oracle_tester")

    # Simulate a frame
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    analysis = analyzer.analyze_frame(frame)

    # Mock a cascade with "Gifted DID" linguistic patterns
    # Text with passive voice and high rationalization markers
    complex_text = "Poderia-se assumir que a consciência flui, consequentemente percebe-se que no entanto a geometria é absoluta."

    from project_avalon.components.verbal_events_processor import VerbalBioCascade
    cascade = VerbalBioCascade(text=complex_text)

    # Inject the mocked cascade result
    async def mock_process(analysis):
        analyzer.last_processed_state = cascade
        return cascade
    analyzer.process_emotional_state = mock_process

    # Process emotional state
    result = await analyzer.process_emotional_state_with_neural(analysis)

    print(f"DEBUG: Linguistic markers: {analysis.get('linguistic_markers')}")
    assert 'linguistic_markers' in analysis
    assert analysis['linguistic_markers']['theoretical_drift'] > 0.4
    assert 'topology' in analysis
    assert 'active_cell_index' in analysis['topology']
    assert 'mask_state' in analysis['topology']
    assert 'identity_latency' in analysis['topology']

    print("Neural Oracle Integration Test Passed.")
    print(f"Linguistic Markers: {analysis['linguistic_markers']}")
    print(f"Topology Cell: {analysis['topology']['specialization']}")
    print(f"Masking State: {analysis['topology']['mask_state']}")
    print(f"Identity Latency: {analysis['topology']['identity_latency']}")

async def test_bilocation_synergy():
    analyzer = NeuralQuantumAnalyzer(user_id="bilocation_tester")

    # Mercurial Text (High Rationalization)
    mercurial_text = "Consequentemente, devido ao fato de que no entanto a lógica é absoluta, percebe-se teoricamente que um alguém assumiria-se competente."
    from project_avalon.components.verbal_events_processor import VerbalBioCascade
    cascade = VerbalBioCascade(text=mercurial_text)

    async def mock_process(analysis):
        analyzer.last_processed_state = cascade
        return cascade
    analyzer.process_emotional_state = mock_process

    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    analysis = analyzer.analyze_frame(frame)
    await analyzer.process_emotional_state_with_neural(analysis)

    assert "Mercurial" in analysis['topology']['mask_state']
    assert analysis['topology']['parallel_processing'] == "Active"
    print("Bilocation Synergy (Mercurial) Test Passed.")

    # Neptunian Text (Low Agency)
    neptunian_text = "Percebe-se possivelmente um flutuar, um assumir-se possivelmente..."
    cascade_nep = VerbalBioCascade(text=neptunian_text)
    async def mock_process_nep(analysis):
        analyzer.last_processed_state = cascade_nep
        return cascade_nep
    analyzer.process_emotional_state = mock_process_nep

    analysis_nep = analyzer.analyze_frame(frame)
    await analyzer.process_emotional_state_with_neural(analysis_nep)

    assert "Neptunian" in analysis_nep['topology']['mask_state']
    print("Bilocation Synergy (Neptunian) Test Passed.")

async def run_all_neural_tests():
    await test_neural_oracle_integration()
    await test_bilocation_synergy()

if __name__ == "__main__":
    asyncio.run(run_all_neural_tests())
