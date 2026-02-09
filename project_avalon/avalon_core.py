# project_avalon/avalon_core.py
import sys
import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from queue import Queue
import json
from datetime import datetime
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_avalon.quantum.grover_neural_search import GroverNeuralSearch, NeuralPattern
from project_avalon.sensors.bioelectric_impedance import BioelectricImpedanceSensor
from project_avalon.philosophy.arkhe_core import ArkheCore, ArkhePreservationProtocol
from project_avalon.philosophy.holographic_weaver import HolographicWeaver
from project_avalon.utils.quantum_dns import QuantumDNS
from project_avalon.audio.identity_sound import IdentitySoundGenerator

# Arkhe(n) v5.0 Integrations
from project_avalon.protocols.quantum_sarcophagus import QuantumSarcophagus
from project_avalon.audio.cosmic_jam import CosmicJamSession
from project_avalon.audio.pentalogy_melody import PentalogyMelody
from project_avalon.quantum.echo_receiver import SaturnEchoScanner

@dataclass
class EEGMetrics:
    alpha: float = 0.0
    beta: float = 0.0
    theta: float = 0.0
    gamma: float = 0.0
    coherence: float = 0.0

    @property
    def focus_score(self) -> float:
        """Score de 0-1 para foco/atenção"""
        return np.clip(self.beta / (self.theta + 0.001), 0, 2) / 2

    @property
    def calm_score(self) -> float:
        """Score de 0-1 para relaxamento"""
        return np.clip(self.alpha / (self.beta + 0.001), 0, 2) / 2

    def calculate_entropy(self) -> float:
        """Calcula a Entropia de Shannon (S) das bandas EEG."""
        # Normalizar as potências para formar uma distribuição de probabilidade
        powers = np.array([self.alpha, self.beta, self.theta, self.gamma])
        total_power = np.sum(powers) + 1e-9
        p = powers / total_power
        # S = -sum(p * log(p))
        entropy = -np.sum(p * np.log(p + 1e-9))
        # Normalizar para 0-1 (ln(4) ≈ 1.386)
        return float(np.clip(entropy / 1.386, 0, 1))

class AvalonKalkiSystem:
    """
    Núcleo do sistema Avalon v5.0 (Pentad Integration).
    Integra Arkhé (A), Biologia (B), Campo (C), DNA (D) e Transcendência (E).
    """

    def __init__(self):
        self.is_running = False
        self.metrics = EEGMetrics()
        self.session_data = []
        self.metrics_history = []
        self.start_time = None

        # AQFI & Arkhe Components
        self.grover_search = GroverNeuralSearch(backend='classical')
        self.impedance_sensor = BioelectricImpedanceSensor()
        self.user_arkhe = ArkheCore.generate_from_identity("default_user")
        self.arkhe_preservation = ArkhePreservationProtocol(self.user_arkhe)
        self.holographic_weaver = HolographicWeaver(self.user_arkhe)

        # New Integrated Modules
        self.dns_resolver = QuantumDNS()
        self.sound_key_gen = IdentitySoundGenerator(self.user_arkhe)

        # Arkhe(n) v5.0 Protocols
        self.sarcophagus = QuantumSarcophagus("Hal Finney")
        self.jam_session = CosmicJamSession()
        self.pentalogy_melody = PentalogyMelody()
        self.echo_receiver = SaturnEchoScanner()

        # Pentad Multiplier (A*B*C*D*E)
        # ABC*D = 4308 hex (17160 dec)
        # ABC*D*E (E=14) = 240240 dec = 3AA70 hex
        self.pentad_seed = 240240

        # Kalki SOC (Self-Organized Criticality) Model
        self.soc_grid = np.zeros((20, 20)) # Pile of sand grains (neural stress)
        self.soc_threshold = 4
        self.yuga_state = "Satya" # Satya, Treta, Dvapara, Kali
        self.malleability_score = 0.5

        # Sistema de módulos
        self.modules = {
            'visual': None,
            'audio': None,
            'hardware': None,
            'protocol': None
        }

        # Comunicação entre módulos
        self.message_queue = Queue()

    def bootstrap(self):
        """Inicialização automática de todos os módulos"""
        print("🚀 Bootstrapping Avalon System v3.1 (Holographic Mesh Enabled)...")

        # 1. Resolve System Node via Quantum DNS
        self.node_id = self.dns_resolver.resolve('avalon.asi')

        # 2. Inicializar módulos
        self._init_hardware()
        self._init_protocol()
        self._init_audio()
        self._init_visual()

        # 3. Apply Identity Sound Key to Audio Engine
        if self.modules['audio']:
            key_freq = self.sound_key_gen.generate_key_frequency()
            self.modules['audio'].set_frequency(key_freq)
            print(f"   [AUDIO] Identity Sound Key synchronized: {key_freq:.2f}Hz")

        print("✅ Sistema inicializado e sincronizado no Campo")
        return True

    def _init_visual(self):
        """Inicializa visualização OpenGL Quaternária (v5.0)"""
        try:
            from project_avalon.visual.quaternary_viz import QuaternaryViz
            self.modules['visual'] = QuaternaryViz()
            return True
        except Exception as e:
            print(f"   ⚠️  Visualização: {e}")
            return False

    def _init_audio(self):
        """Inicializa sistema de áudio"""
        try:
            from project_avalon.audio.low_latency_feedback import AudioEngine
            audio = AudioEngine()
            audio.start()
            self.modules['audio'] = audio
            return True
        except Exception as e:
            print(f"   ⚠️  Áudio: {e}")
            return False

    def _init_hardware(self):
        """Tenta conectar ao hardware EEG"""
        try:
            from project_avalon.hardware.universal_eeg import UniversalEEG
            eeg = UniversalEEG()
            eeg.auto_connect()
            self.modules['hardware'] = eeg
            return True
        except Exception as e:
            print(f"   ❌ Hardware: {e}")
            return False

    def _init_protocol(self):
        """Carrega protocolos terapêuticos"""
        try:
            from project_avalon.protocols.manager import ProtocolManager
            self.modules['protocol'] = ProtocolManager()
            return True
        except Exception as e:
            print(f"   ⚠️  Protocolos: {e}")
            return False

    def start_session(self, protocol_name: str = "flow", duration: int = 60):
        """Inicia uma sessão completa com busca quântica Grover"""
        print(f"\n🚀 INICIANDO SESSÃO ASI: {protocol_name} ({duration}s)")
        print(f"   Princípio: 1A × 2B = 45E | Speedup Quântico: {self.grover_search.find_closest_ideal({})['quantum_speedup']:.1f}x")
        print("=" * 60)

        self.is_running = True
        self.start_time = time.time()
        self.session_data = []
        self.metrics_history = []

        # Configurar protocolo
        protocol_obj = self.modules['protocol'].get_protocol(protocol_name)

        # Loop principal
        try:
            last_quantum_search = 0
            last_weave = 0
            while self.is_running and (time.time() - self.start_time) < duration:
                current_time = time.time()

                # 1. Coletar dados
                metrics_raw = self._collect_data()

                # 2. Busca Quântica Periódica (a cada 5s)
                if current_time - last_quantum_search > 5.0:
                    quantum_result = self.grover_search.find_closest_ideal({
                        'coherence': metrics_raw.coherence,
                        'entropy': metrics_raw.calculate_entropy(),
                        'alpha': metrics_raw.alpha,
                        'beta': metrics_raw.beta
                    })
                    print(f"\n⚛️ [GROVER SEARCH] Padrão Ideal Detectado (Prob: {quantum_result['search_result']['probability']:.1%})")
                    last_quantum_search = current_time

                # 3. Processar com protocolo
                feedback = protocol_obj.process(metrics_raw)

                # 4. Aplicar feedback
                self._apply_feedback(feedback, metrics_raw)

                # 5. Verificar Protocolo de Reset Kalki (ASI-enhanced)
                self.kalki_reset_protocol(metrics_raw)

                # 6. Tecelão Holográfico (Cura por Redundância) - a cada 8s
                if current_time - last_weave > 8.0:
                    print("\n🧶 [TECEDOR] Sincronizando campo morfogenético...")
                    # Simula um manifold de 256 dimensões
                    manifold = np.random.rand(256)
                    self.holographic_weaver.weave_identity(manifold)
                    # Play Identity Sound Key
                    if self.modules['audio']:
                        self.modules['audio'].set_frequency(self.holographic_weaver.get_identity_key())
                    last_weave = current_time

                # 7. Logging
                self._log_frame(metrics_raw, feedback)
                self.metrics_history.append({'metrics': metrics_raw.__dict__})

                # 7. Pequena pausa
                time.sleep(0.05)  # 20Hz

        except KeyboardInterrupt:
            print("\n⏹️  Sessão interrompida pelo usuário")

        # Finalizar
        self.stop_session()
        return self._generate_report()

    def _collect_data(self) -> EEGMetrics:
        """Coleta dados de todas as fontes"""
        if self.modules['hardware']:
            raw = self.modules['hardware'].get_metrics()
            self.metrics = EEGMetrics(
                alpha=raw.get('alpha', 0.5),
                beta=raw.get('beta', 0.3),
                theta=raw.get('theta', 0.2),
                gamma=raw.get('gamma', 0.1),
                coherence=raw.get('coherence', 0.6)
            )
        return self.metrics

    def _apply_feedback(self, feedback: Dict, metrics: EEGMetrics):
        """Aplica feedback a todos os módulos"""
        # Visual
        if self.modules['visual']:
            # Call from main thread if using GUI
            try:
                # Assuming the visualizer window has an update_metrics method
                self.modules['visual'].update_metrics({
                    'coherence': metrics.coherence,
                    'focus': metrics.focus_score
                })
            except Exception as e:
                pass

        # Áudio
        if self.modules['audio']:
            self.modules['audio'].set_frequency(feedback.get('audio_frequency', 440))

    def _log_frame(self, metrics: EEGMetrics, feedback: Dict):
        """Registra frame atual"""
        frame = {
            'timestamp': time.time() - (self.start_time or 0),
            'metrics': {
                'alpha': metrics.alpha,
                'beta': metrics.beta,
                'theta': metrics.theta,
                'gamma': metrics.gamma,
                'coherence': metrics.coherence,
                'focus': metrics.focus_score,
                'calm': metrics.calm_score
            },
            'feedback': feedback
        }
        self.session_data.append(frame)

        # Log simples no console
        if len(self.session_data) % 20 == 0:  # A cada segundo
            print(f"⏱️  {len(self.session_data)/20:.0f}s | "
                  f"Foco: {metrics.focus_score:.2f} | "
                  f"Calma: {metrics.calm_score:.2f}")

    def update_soc_model(self, metrics: EEGMetrics):
        """Atualiza o modelo de Pilha de Areia (SOC) com o estresse neural"""
        # Adiciona 'grãos' de estresse baseados na entropia
        entropy = metrics.calculate_entropy()
        num_grains = int(entropy * 5)
        for _ in range(num_grains):
            x, y = np.random.randint(0, 20, 2)
            self.soc_grid[x, y] += 1

        # Toppling (Avalanches)
        while np.any(self.soc_grid >= self.soc_threshold):
            x, y = np.where(self.soc_grid >= self.soc_threshold)
            for i, j in zip(x, y):
                self.soc_grid[i, j] -= 4
                if i > 0: self.soc_grid[i-1, j] += 1
                if i < 19: self.soc_grid[i+1, j] += 1
                if j > 0: self.soc_grid[i, j-1] += 1
                if j < 19: self.soc_grid[i, j+1] += 1

        # Yuga Detection based on grid mass
        total_mass = np.sum(self.soc_grid)
        if total_mass < 200: self.yuga_state = "Satya"
        elif total_mass < 500: self.yuga_state = "Treta"
        elif total_mass < 800: self.yuga_state = "Dvapara"
        else: self.yuga_state = "Kali"

    def kalki_reset_protocol(self, metrics: EEGMetrics):
        """
        Protocolo de Segurança KALKI v2.0
        Finalidade: Interromper a criticalidade auto-organizada destrutiva.
        """
        self.update_soc_model(metrics)
        entropy = metrics.calculate_entropy()
        coherence = metrics.coherence

        # ASI enhancement: Check substrate malleability
        substrate_info = self.impedance_sensor.measure_substrate_malleability(entropy, coherence)
        self.malleability_score = substrate_info['malleability_score']

        # Trigger Condition: Kali Yuga State OR Extreme Entropy/Rigidity
        if self.yuga_state == "Kali" or (entropy > 0.9 and coherence < 0.1) or self.malleability_score < 0.2:
            print(f"\n🚨 [KALKI RESET v2.0] Criticalidade SOC: {self.yuga_state} Yuga detectado")
            self.execute_kalki_strike()

    def activate_sarcophagus(self, dna_sample: Optional[str] = None):
        """
        Ativa o Sarcófago de Informação Quântica.
        Fragmenta o DNA para imortalidade via blockchain.
        """
        print("\n🧬 [ARKHE(N)] Ativando Sarcófago de Informação Quântica...")
        dna = dna_sample or self.sarcophagus.generate_genome_sample(320)
        fragments = self.sarcophagus.fragment_genome(dna)
        genesis = self.sarcophagus.get_genesis_signature()

        print(f"   Sujeito: {genesis['subject']}")
        print(f"   Protocolo: {genesis['protocol']}")
        print(f"   Fragmentos Gerados: {len(fragments)}")
        print(f"   Assinatura Entrópica: {fragments[0]['entropy']:.4f} (Biossinal Detectado)")

        # Simula a 'injeção' na blockchain
        for frag in fragments[:2]:
            print(f"   [BLOCKCHAIN] Injetando OP_RETURN: {frag['hex'][:30]}... (OK)")

        return fragments

    def initiate_jam_session(self):
        """Inicia uma Jam Session Cósmica com Saturno e Enceladus"""
        print("\n🎵 [ARKHE(N)] Iniciando Jam Session Cósmica...")
        cure = self.jam_session.generate_enceladus_cure()
        framework = self.jam_session.get_jam_framework()

        print(f"   Protocolo de Cura: {cure['title']}")
        print(f"   Frequências: {cure['base_frequencies']} Hz")

        if self.modules['audio']:
            # Tocar a frequência de Enceladus
            self.modules['audio'].set_frequency(cure['base_frequencies'][0])
            time.sleep(1)
            self.modules['audio'].set_frequency(cure['base_frequencies'][1])

        print(f"   Participantes: {', '.join(framework['participants'].keys())}")
        print("   Status: Jam Session Sincronizada com o Campo")

    def transmit_final_melody(self):
        """
        Sintetiza e transmite a Melodia Final ABCDE via gateway 0.0.0.0.
        Sela a singularidade pentalógica (3AA70).
        """
        print("\n🎵 [ARKHE(N)] Sintetizando Melodia Final ABCDE (v5.0 Transcendence)...")
        melody = self.pentalogy_melody.synthesize()
        packets = self.pentalogy_melody.encode_for_gateway(melody)

        print(f"   [PENTAD] Sincronização Estabilizada: 3AA70 hex")
        print(f"   [GATEWAY] Transmitindo {len(packets)} pacotes quânticos para o ano 12.024...")

        if self.modules['audio']:
            # Simula a transmissão sonora (usando a frequência de transcendência)
            self.modules['audio'].set_frequency(240240.0 / 1000) # Audível

        print("   [ARKHE(N)] Colapso de Onda Temporal Completo. O Manifold é Atemporal.")
        return packets

    def scan_future_echoes(self):
        """Realiza varredura de ecos do futuro (ano 12.024)"""
        print("\n📡 [ARKHE(N)] Varrendo gateway 0.0.0.0 para Ecos do Futuro...")
        result = self.echo_receiver.scan_future_resonance()
        print(f"   Echo-Block Detectado: {result['echo_block_id']}")
        print(f"   Origem Temporal: Ano {result['timestamp_future']}")
        print(f"   Mensagem do Futuro: {result['thought_patterns'][4]}")

        # Aplicar PoBF Decodificado
        instruction = self.echo_receiver.decode_echo_block(result['echo_block_id'])
        print(f"   [DECODIFICADOR] {instruction}")

        return result

    def execute_kalki_strike(self):
        """A 'Espada' (Pattern Interruption), o 'Cavalo' (Solfeggio) e o 'Satya' (Schumann)."""
        print("⚔️  A ESPADA: Interrupção súbita de frequências dissonantes")
        if self.modules['audio']:
            self.modules['audio'].set_frequency(880) # O 'Grito' de Kalki
        if self.modules['visual']:
            try: self.modules['visual'].trigger_kalki_flash()
            except: pass
        time.sleep(0.5)

        print("🌀 O CAVALO BRANCO: Indução de Frequências Solfeggio para Cura")
        solfeggio = [174, 285, 396, 417, 528, 639, 741, 852]
        for freq in solfeggio:
            if self.modules['audio']:
                self.modules['audio'].set_frequency(freq)
            time.sleep(0.2) # Rapid sweep

        print("⚖️  SATYA YUGA: Reestabelecendo Dharma (Ressonância de Schumann 7.83Hz)")
        if self.modules['audio']:
            self.modules['audio'].set_frequency(7.83)

        # Arkhe preservation check
        result = self.arkhe_preservation.execute_safe_reset(intensity=0.8)
        print(f"   [ARKHE] Integridade de Identidade: {result['status']} ({result['integrity']:.2f})")

        # Reset SOC Grid
        self.soc_grid = np.zeros((20, 20))
        time.sleep(1.0)

    def stop_session(self):
        """Para a sessão atual"""
        self.is_running = False

        # Parar todos os módulos
        if self.modules['audio']:
            self.modules['audio'].stop()

    def _generate_report(self) -> Dict:
        """Gera relatório da sessão"""
        if not self.session_data:
            return {}

        # Análise básica
        focus_scores = [f['metrics']['focus'] for f in self.session_data]
        calm_scores = [f['metrics']['calm'] for f in self.session_data]

        report = {
            'duration': time.time() - (self.start_time or 0),
            'frames': len(self.session_data),
            'avg_focus': np.mean(focus_scores),
            'avg_calm': np.mean(calm_scores),
            'max_focus': np.max(focus_scores),
            'max_calm': np.max(calm_scores)
        }

        # Salvar dados JSON
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = os.path.join("project_avalon", "sessions", f"session_{session_id}.json")
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)
        with open(json_filename, 'w') as f:
            json.dump({
                'metadata': report,
                'data': self.session_data
            }, f, indent=2)

        # Salvar dados CSV (usando pandas)
        try:
            import pandas as pd
            flat_data = []
            for frame in self.session_data:
                row = {'timestamp': frame['timestamp']}
                row.update(frame['metrics'])
                row.update(frame['feedback'])
                flat_data.append(row)

            df = pd.DataFrame(flat_data)
            csv_filename = os.path.join("project_avalon", "sessions", f"session_{session_id}.csv")
            df.to_csv(csv_filename, index=False)
            print(f"📊 Relatório CSV salvo em: {csv_filename}")
        except ImportError:
            print("⚠️ Pandas não instalado. Exportação CSV pulada.")

        print(f"📊 Relatório JSON salvo em: {json_filename}")
        return report

# Global Alias
AvalonCore = AvalonKalkiSystem

def main():
    """Função principal (CLI)"""
    print("""
    ╔══════════════════════════════════════╗
    ║        AVALON NEUROFEEDBACK         ║
    ║     Sistema Prático v3.0 (AQFI)    ║
    ║     Princípio: 1A × 2B = 45E      ║
    ╚══════════════════════════════════════╝
    """)

    system = AvalonKalkiSystem()
    system.bootstrap()

    # Menu interativo
    while True:
        print("\n" + "="*50)
        print("MENU PRINCIPAL")
        print("="*50)
        print("1. Sessão de Foco (1 min)")
        print("2. Sessão de Calma (1 min)")
        print("3. Sessão de Flow (1 min)")
        print("4. Ver módulos carregados")
        print("5. Sair")

        choice = input("\nEscolha: ").strip()

        if choice == '1':
            system.start_session('focus', 60)
        elif choice == '2':
            system.start_session('calm', 60)
        elif choice == '3':
            system.start_session('flow', 60)
        elif choice == '4':
            print("\n📦 Módulos carregados:")
            for name, module in system.modules.items():
                status = "✅" if module else "❌"
                print(f"   {status} {name}")
        elif choice == '5':
            break

if __name__ == "__main__":
    main()
