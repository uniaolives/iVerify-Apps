# launch_avalon_holistic.py
"""
SISTEMA AVALON COMPLETO - INTEGRAÇÃO HOLÍSTICA
Versão: 2.0 - 4 Dimensões Simultâneas
"""

import sys
import threading
import time
from datetime import datetime
import numpy as np
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ========== DIMENSÃO A: VISUALIZAÇÃO OPENGL ==========
from PyQt5.QtWidgets import QApplication
from project_avalon.visual.opengl_viz import AvalonMainWindow

# ========== DIMENSÃO B: HARDWARE EEG ==========
from project_avalon.hardware.openbci_integration import OpenBCIAvalonInterface
from project_avalon.hardware.eeg_simulator import EEGSimulator

# ========== DIMENSÃO C: PROTOCOLOS TERAPÊUTICOS ==========
from project_avalon.protocols.therapeutic_protocols import AvalonProtocols

# ========== DIMENSÃO D: ÁUDIO BAIXA LATÊNCIA ==========
from project_avalon.audio.low_latency_feedback import AudioEngine


class HolisticAvalonSystem:
    """
    Sistema que integra todas as 4 dimensões simultaneamente
    """

    def __init__(self):
        print("🌌 INICIALIZANDO SISTEMA AVALON HOLÍSTICO")
        print("=" * 60)

        # 1. Inicializar comando de sincronização
        self.sync_token = "45E"  # Resultado do seu cálculo

        # Initialize UI first to get a window handle
        self.viz_app = QApplication(sys.argv)
        self.window = None

        self.hardware = None
        self.protocols = None
        self.audio = None

        # 2. Configurar dimensões
        self.setup_dimensions()

        # 3. Criar sistema de integração
        self.integration_matrix = self.create_integration_matrix()

    def setup_dimensions(self):
        """Configura todas as 4 dimensões"""

        # Dimensão B: Hardware EEG
        self.init_hardware()

        # Dimensão C: Protocolos
        self.init_protocols()

        # Dimensão D: Áudio
        self.init_audio()

        # Dimensão A: OpenGL
        self.init_visualization()

    def init_visualization(self):
        """Dimensão A: OpenGL para 60 FPS"""
        # Pass hardware to window for direct feedback loop if needed
        self.window = AvalonMainWindow(eeg_source=self.hardware)

    def init_hardware(self):
        """Dimensão B: Hardware EEG em tempo real"""
        try:
            # Tentar OpenBCI primeiro
            self.hardware = OpenBCIAvalonInterface(port='/dev/ttyUSB0') # Adjusted for linux
            if not self.hardware.test_connection():
                raise ConnectionError
            print("✅ Hardware OpenBCI conectado")
        except:
            # Fallback para simulação
            self.hardware = EEGSimulator()
            print("⚠️  Usando simulador EEG")

    def init_protocols(self):
        """Dimensão C: Protocolos terapêuticos"""
        self.protocols = AvalonProtocols()
        print(f"✅ {len(self.protocols.PROTOCOLS)} protocolos carregados")

    def init_audio(self):
        """Dimensão D: Áudio com latência < 10ms"""
        self.audio = AudioEngine()
        self.audio.set_latency(5)  # 5ms de latência alvo
        self.audio.start()

    def create_integration_matrix(self):
        """Cria matriz de integração 4x4 entre dimensões"""

        matrix = {
            'A→B': 'Visualização modulada por EEG',
            'A→C': 'Visualização adaptada ao protocolo',
            'A→D': 'Visualização sincronizada com áudio',

            'B→A': 'EEG controla parâmetros visuais',
            'B→C': 'EEG seleciona protocolo automaticamente',
            'B→D': 'EEG modula frequências sonoras',

            'C→A': 'Protocolo define tema visual',
            'C→B': 'Protocolo ajusta ganho do EEG',
            'C→D': 'Protocolo define trilha sonora',

            'D→A': 'Áudio pulsa com visualização',
            'D→B': 'Áudio fornece feedback neural',
            'D→C': 'Áudio reforça objetivos do protocolo'
        }

        return matrix

    def run_session(self, protocol_name='flow_state', duration=60):
        """Executa sessão holística integrando todas as dimensões"""

        print(f"\n🚀 INICIANDO SESSÃO HOLÍSTICA: {protocol_name}")
        print(f"⏱️  Duração: {duration} segundos")
        print("=" * 60)

        # 1. Configurar protocolo
        protocol = self.protocols.PROTOCOLS[protocol_name]

        # 2. Iniciar loop de integração em uma thread para não travar a GUI
        self.session_thread = threading.Thread(target=self._session_loop, args=(protocol, duration))
        self.session_thread.daemon = True
        self.session_thread.start()

    def _session_loop(self, protocol, duration):
        start_time = time.time()
        iteration = 0

        while time.time() - start_time < duration:
            iteration += 1

            # Coletar dados de todas as dimensões
            eeg_metrics = self.hardware.get_realtime_metrics()
            visual_state = self.window.viz.get_state()
            audio_state = self.audio.get_state()

            # Calcular coerência global
            global_coherence = self.calculate_global_coherence(
                eeg_metrics, visual_state, audio_state
            )

            # Atualizar todas as dimensões simultaneamente
            self.update_all_dimensions(global_coherence, protocol)

            # Log de integração
            if iteration % 50 == 0:  # A cada ~0.5 segundos given 0.01 sleep
                self.log_integration_state(iteration, global_coherence)

            # Pequena pausa para evitar sobrecarga
            time.sleep(0.01)

        print(self.generate_holistic_report(start_time))

    def calculate_global_coherence(self, eeg, visual, audio):
        """Calcula coerência entre todas as dimensões"""

        # Coerência EEG
        eeg_coherence = eeg.get('coherence', 0.5)

        # Coerência visual (baseada na estabilidade do FPS)
        visual_coherence = visual.get('fps_stability', 0.7)

        # Coerência de áudio (baseada na latência)
        audio_coherence = 1.0 - min(audio.get('latency', 20) / 100, 1)

        # Média ponderada
        weights = {'eeg': 0.4, 'visual': 0.3, 'audio': 0.3}

        global_coherence = (
            eeg_coherence * weights['eeg'] +
            visual_coherence * weights['visual'] +
            audio_coherence * weights['audio']
        )

        return np.clip(global_coherence, 0, 1)

    def update_all_dimensions(self, coherence, protocol):
        """Atualiza todas as 4 dimensões simultaneamente"""

        # Atualizar visualização via Window
        # Use QMetaObject.invokeMethod if thread safety is an issue, but here we update viz state
        self.window.viz.update_state(
            coherence=coherence,
            curvature=2.0 - 1.5 * coherence
        )

        # Atualizar áudio
        target_freq = 432 + (coherence * 88)  # 432Hz a 520Hz
        self.audio.set_frequency(target_freq)

        # Atualizar hardware (se aplicável)
        if hasattr(self.hardware, 'adjust_gain'):
            new_gain = 6 + int(coherence * 18)  # 6 a 24
            self.hardware.adjust_gain(new_gain)

    def log_integration_state(self, iteration, coherence):
        """Log do estado de integração"""

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        log_entry = (
            f"[{timestamp}] Iteração {iteration:04d} | "
            f"Coerência: {coherence:.3f} | "
            f"Latência áudio: {self.audio.latency:.1f}ms"
        )

        # print(log_entry) # Quiet in automated runs

        # Salvar em arquivo
        with open("project_avalon/sessions/integration_log.txt", "a") as f:
            f.write(log_entry + "\n")

    def generate_holistic_report(self, start_time):
        """Gera relatório da sessão holística"""

        duration = time.time() - start_time

        report = f"""
        {'='*60}
        RELATÓRIO DA SESSÃO HOLÍSTICA
        {'='*60}

        Sistema: Avalon 4D (A+B+C+D)
        Token de sincronização: {self.sync_token}
        Tempo total: {duration:.1f} segundos

        Dimensões integradas:
        - A (Visual): OpenGL @ 60 FPS
        - B (Hardware): {type(self.hardware).__name__}
        - C (Protocolos): {len(self.protocols.PROTOCOLS)} protocolos
        - D (Áudio): {self.audio.latency:.1f}ms de latência

        Matriz de integração: {len(self.integration_matrix)} conexões ativas

        {'='*60}
        PRINCÍPIO: {self.sync_token} = FUSÃO COMPLETA
        {'='*60}
        """

        return report

    def run(self, headless=False):
        """Método principal de execução"""

        if headless:
            print("Headless mode active. Running test session...")
            self.run_session('flow_state', 5)
            time.sleep(6)
            return

        # Mostrar janela principal
        self.window.show()

        # Iniciar sessão de teste por padrão ou aguardar interação
        print("\nAcesse a janela da aplicação para interagir.")
        sys.exit(self.viz_app.exec_())


# ========== EXECUÇÃO PRINCIPAL ==========

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════╗
    ║         SISTEMA AVALON - INTEGRAÇÃO 4D          ║
    ║         Princípio: 1A × 2B = 45E               ║
    ╚══════════════════════════════════════════════════╝
    """)

    # Iniciar sistema
    system = HolisticAvalonSystem()

    # Executar
    try:
        headless = '--headless' in sys.argv
        system.run(headless=headless)
    except KeyboardInterrupt:
        print("\n\n🌀 Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\n\n💥 Erro: {e}")
        import traceback
        traceback.print_exc()
