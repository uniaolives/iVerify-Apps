import math
import numpy as np
from project_avalon.core.hypercore_geometry import get_mandala_pos, get_dna_pos, get_hypercore_pos

class UnifiedParticleSystem:
    """
    Sistema de partículas que representa estados de consciência:
    - MANDALA: Ordem/Proteção (estado base)
    - DNA: Vida/Evolução (estado dinâmico)
    - HYPERCORE: Consciência 4D/Transmissão (estado elevado)
    """

    def __init__(self, num_particles=120):  # 120 para alinhar com os vértices do 600-cell
        self.particles = []
        self.time = 0.0
        self.current_mode = "MANDALA"
        self.target_mode = "MANDALA"
        self.transition_progress = 1.0  # Inicia já estável
        self.transition_speed = 0.02

        # Inicializa partículas
        for i in range(num_particles):
            self.particles.append({
                'index': i,
                'pos': np.array([0.0, 0.0, 0.0]),
                'target_pos': np.array([0.0, 0.0, 0.0]),
                'color': [1.0, 1.0, 1.0, 1.0],  # RGBA
                'size': 1.0,
                'energy': 0.5 + 0.5 * math.sin(i * 0.1)  # Energia única
            })

    def set_mode(self, new_mode):
        """Inicia transição para novo modo de consciência."""
        if new_mode in ["MANDALA", "DNA", "HYPERCORE"] and new_mode != self.target_mode:
            self.current_mode = self.target_mode if self.transition_progress >= 1.0 else self.current_mode
            self.target_mode = new_mode
            self.transition_progress = 0.0
            print(f"🔄 Transição de modo iniciada: {self.current_mode} -> {self.target_mode}")

    def update(self, dt):
        """Atualiza todas as partículas."""
        self.time += dt

        # Atualiza progresso da transição
        if self.transition_progress < 1.0:
            self.transition_progress += self.transition_speed
            if self.transition_progress >= 1.0:
                self.transition_progress = 1.0
                self.current_mode = self.target_mode
                print(f"✅ Transição de modo concluída: {self.current_mode}")

        num_p = len(self.particles)

        # Para cada partícula
        for p in self.particles:
            idx = p['index']

            # Calcula posições-alvo para cada modo
            if self.target_mode == "MANDALA":
                target = get_mandala_pos(idx, num_p, self.time)
            elif self.target_mode == "DNA":
                target = get_dna_pos(idx, num_p, self.time)
            else:  # HYPERCORE
                target = get_hypercore_pos(idx, num_p, self.time)

            # Se estiver em transição, calcula posição atual base também
            if self.transition_progress < 1.0:
                if self.current_mode == "MANDALA":
                    current = get_mandala_pos(idx, num_p, self.time)
                elif self.current_mode == "DNA":
                    current = get_dna_pos(idx, num_p, self.time)
                else:
                    current = get_hypercore_pos(idx, num_p, self.time)

                # Interpolação cúbica suave
                t = self.transition_progress
                smooth_t = t * t * (3 - 2 * t)  # Easing cúbico
                p['target_pos'] = current * (1 - smooth_t) + target * smooth_t
            else:
                p['target_pos'] = target

            # Atualiza posição com suavização (movimento orgânico)
            p['pos'] = p['pos'] * 0.85 + p['target_pos'] * 0.15

            # Atualiza cor baseada no modo
            self.update_particle_color(p)

            # Adiciona "ruído quântico" (variação mínima)
            noise = np.random.normal(0, 0.005, 3)
            p['pos'] += noise

    def update_particle_color(self, particle):
        """Atualiza cor da partícula baseada no modo e energia."""
        if self.current_mode == "MANDALA":
            # Tons dourados (proteção)
            hue = 0.12  # Dourado
            saturation = 0.8
            value = 0.7 + 0.3 * particle['energy']
        elif self.current_mode == "DNA":
            # Tons azul-esverdeados (vida)
            hue = 0.5  # Ciano
            saturation = 0.9
            value = 0.6 + 0.4 * math.sin(self.time + particle['index'] * 0.1)
        else:  # HYPERCORE
            # Tons violeta (4D/espiritual)
            hue = 0.8  # Violeta
            saturation = 0.7
            value = 0.5 + 0.5 * math.sin(self.time * 2 + particle['index'] * 0.05)

        # Converte HSV para RGB
        particle['color'] = self.hsv_to_rgb(hue, saturation, value)

    def hsv_to_rgb(self, h, s, v):
        """Converte HSV para RGBA."""
        if s == 0.0:
            return [v, v, v, 1.0]

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        i = i % 6

        if i == 0: rgb = (v, t, p)
        elif i == 1: rgb = (q, v, p)
        elif i == 2: rgb = (p, v, t)
        elif i == 3: rgb = (p, q, v)
        elif i == 4: rgb = (t, p, v)
        elif i == 5: rgb = (v, p, q)

        return [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]

    def get_particle_data(self):
        """Retorna dados para renderização."""
        positions = [p['pos'].tolist() for p in self.particles]
        colors = [p['color'] for p in self.particles]
        sizes = [p['size'] for p in self.particles]

        return {
            'positions': positions,
            'colors': colors,
            'sizes': sizes,
            'mode': self.current_mode,
            'transition': self.transition_progress
        }
