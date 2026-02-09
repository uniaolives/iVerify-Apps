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

class AvalonCore:
    """
    Núcleo do sistema Avalon - Integra tudo em um só lugar
    Versão mínima e funcional
    """

    def __init__(self):
        self.is_running = False
        self.metrics = EEGMetrics()
        self.session_data = []
        self.start_time = None

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
        print("🚀 Bootstrapping Avalon System...")

        # 1. Inicializar módulos em paralelo (conceitualmente)
        self._init_hardware()
        self._init_protocol()
        self._init_audio()
        # Visualizer often needs main thread, but here we just prepare it

        print("✅ Sistema inicializado")
        return True

    def _init_visual(self):
        """Inicializa visualização OpenGL"""
        try:
            from project_avalon.visual.opengl_viz import NeuroVizWindow
            self.modules['visual'] = NeuroVizWindow()
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
        """Inicia uma sessão completa"""
        print(f"\n🎯 Iniciando sessão: {protocol_name} ({duration}s)")
        print("=" * 50)

        self.is_running = True
        self.start_time = time.time()
        self.session_data = []

        # Configurar protocolo
        protocol = self.modules['protocol'].get_protocol(protocol_name)

        # Loop principal
        try:
            while self.is_running and (time.time() - self.start_time) < duration:
                # 1. Coletar dados
                metrics_raw = self._collect_data()

                # 2. Processar com protocolo
                feedback = protocol.process(metrics_raw)

                # 3. Aplicar feedback
                self._apply_feedback(feedback, metrics_raw)

                # 4. Verificar Protocolo de Reset Kalki
                self.kalki_reset_protocol(metrics_raw)

                # 5. Logging
                self._log_frame(metrics_raw, feedback)

                # 6. Pequena pausa
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

    def kalki_reset_protocol(self, metrics: EEGMetrics):
        """
        Protocolo de Segurança: RESET KALKI
        Finalidade: Interromper loops de feedback de ansiedade e forçar
        uma transição de fase para repouso profundo.
        """
        entropy = metrics.calculate_entropy()
        coherence = metrics.coherence

        # Condição de Gatilho: Alta Entropia (> 0.85) + Baixa Coerência (< 0.2)
        if entropy > 0.85 and coherence < 0.2:
            print("\n🚨 [KALKI RESET] Singularidade Detectada: Kali Yuga Neural")
            self.execute_kalki_strike()

    def execute_kalki_strike(self):
        """A 'Espada' que corta o ruído e o 'Cavalo' que guia ao Satya."""

        # 1. A ESPADA: Corte súbito via Áudio e Visual
        if self.modules['audio']:
            # Sharp tone to clear the buffer
            self.modules['audio'].set_frequency(880)
            time.sleep(0.2)
            # Transition to Schumann Resonance (7.83Hz simulated in audio engine)
            self.modules['audio'].set_frequency(7.83)

        if self.modules['visual']:
            try:
                # Trigger flash and geometry reset
                self.modules['visual'].trigger_kalki_flash()
            except:
                pass

        print("🌀 Reestabelecendo Dharma: Sincronizando com Ressonância de Schumann (7.83Hz)")
        time.sleep(1.0) # Integration pause

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

def main():
    """Função principal (CLI)"""
    print("""
    ╔══════════════════════════════════════╗
    ║        AVALON NEUROFEEDBACK         ║
    ║     Sistema Prático v1.0           ║
    ║     Princípio: 1A × 2B = 45E      ║
    ╚══════════════════════════════════════╝
    """)

    system = AvalonCore()
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
