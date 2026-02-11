from project_avalon.core.unified_particle_system import UnifiedParticleSystem

class ConsciousnessVisualizer3D:
    """
    Integrador de visualização 3D para estados de consciência.
    Conecta o UnifiedParticleSystem com a interface gráfica e dados EEG.
    """

    def __init__(self, num_particles=120):
        # Sistema de partículas
        self.particle_system = UnifiedParticleSystem(num_particles=num_particles)

        # Controles de interface
        self.modes = ["MANDALA", "DNA", "HYPERCORE"]
        self.current_mode_index = 0

        # Estado do Biofeedback
        self.attention_level = 0.5
        self.meditation_level = 0.5

    def update_from_eeg(self, eeg_data):
        """
        Atualiza visualização baseada em dados EEG reais ou simulados.
        """
        if eeg_data:
            # Assume que eeg_data tem atributos attention e meditation (0-100)
            self.attention_level = getattr(eeg_data, 'attention', 50) / 100.0
            self.meditation_level = getattr(eeg_data, 'meditation', 50) / 100.0

            # Lógica de troca de modo baseada no estado mental
            if self.attention_level > 0.7:
                self.particle_system.set_mode("DNA")
            elif self.meditation_level > 0.7:
                self.particle_system.set_mode("HYPERCORE")
            elif self.attention_level < 0.3 and self.meditation_level < 0.3:
                self.particle_system.set_mode("MANDALA")

    def render_frame(self, dt):
        """
        Gera um frame da visualização. Chamado pelo loop principal da GUI.
        """
        # 1. Atualiza sistema de partículas
        self.particle_system.update(dt)

        # 2. Obtém dados calculados
        data = self.particle_system.get_particle_data()

        # 3. Renderização (Stubs para integração com OpenGL/PyQt5)
        self._render_to_gpu(data)

        return data

    def _render_to_gpu(self, data):
        """
        Ponto de integração com drivers de vídeo (PyOpenGL/PyQtGraph).
        """
        # Aqui seriam chamadas funções como glDrawArrays ou update de scatter plot
        pass

    def render_hypercore_connections(self, positions):
        """
        Renderiza linhas conectando os vértices do Hyper-Core.
        """
        # Lógica para desenhar as arestas do 600-cell projetado
        pass

    def get_hud_data(self):
        """
        Retorna informações para o overlay da interface.
        """
        data = self.particle_system.get_particle_data()
        return {
            'mode': data['mode'],
            'transition': data['transition'],
            'attention': f"{self.attention_level*100:.1f}%",
            'meditation': f"{self.meditation_level*100:.1f}%"
        }
