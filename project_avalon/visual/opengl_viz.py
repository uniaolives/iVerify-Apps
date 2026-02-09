# project_avalon/visual/opengl_viz.py
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QOpenGLWidget, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from OpenGL.GL import *
from OpenGL.GLU import *

class NeuroVizWidget(QOpenGLWidget):
    """Widget OpenGL para visualização neural"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)

        # Estado
        self.vertices = []
        self.edges = []
        self.phase = 0.0
        self.coherence = 0.5
        self.flash_intensity = 0.0

        # Timer de atualização
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_phase)
        self.timer.start(16)  # ~60FPS

    def update_phase(self):
        self.phase += 0.01
        self.update()

    def update_state(self, metrics_dict=None, **kwargs):
        """Atualiza com novas métricas EEG"""
        if metrics_dict is None:
            metrics_dict = kwargs
        self.coherence = metrics_dict.get('coherence', 0.5)
        self.generate_geometry()

    def get_state(self):
        """Returns visual state for integration matrix."""
        return {'fps_stability': 0.95}

    def generate_geometry(self):
        """Gera geometria do manifold neural"""
        phi = (1 + np.sqrt(5)) / 2

        self.vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=np.float32)

        scale = 0.5 + self.coherence * 1.0
        self.vertices *= scale

        self.edges = []
        for i in range(len(self.vertices)):
            for j in range(i+1, len(self.vertices)):
                if np.linalg.norm(self.vertices[i] - self.vertices[j]) < 3.0 * scale:
                    self.edges.append((i, j))

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.1, 1.0)
        self.generate_geometry()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Apply Flash Effect
        if self.flash_intensity > 0:
            glClearColor(self.flash_intensity, self.flash_intensity, self.flash_intensity, 1.0)
            self.flash_intensity *= 0.9 # Decay
            if self.flash_intensity < 0.01:
                self.flash_intensity = 0.0
                glClearColor(0.0, 0.0, 0.1, 1.0) # Back to normal blue

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # O PULSO DO CRISTAL DO TEMPO (Sub-harmonic 24ms)
        # Phase from 0 to 2pi every ~24ms (simulated here by self.phase)
        pulse = 1.0 + 0.2 * np.sin(self.phase * 5)

        glTranslatef(0, 0, -8)
        glRotatef(self.phase * 30, 0, 1, 0)
        glRotatef(self.phase * 15, 1, 0, 0)

        glLineWidth(2.0)
        glBegin(GL_LINES)
        for i, j in self.edges:
            # Cor baseada na coerência
            r = 0.2 + 0.8 * self.coherence
            g = 0.8 - 0.6 * self.coherence
            b = 0.9
            glColor3f(r, g, b)

            glVertex3fv(self.vertices[i] * pulse)
            glVertex3fv(self.vertices[j] * pulse)
        glEnd()

        glPointSize(10.0 + 10.0 * self.coherence)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.9, 0.4)
        for v in self.vertices:
            glVertex3fv(v)
        glEnd()

class NeuroVizWindow(QWidget):
    """Janela principal da visualização"""

    def __init__(self, eeg_source=None):
        super().__init__()
        self.eeg_source = eeg_source
        self.setWindowTitle("Avalon NeuroViz")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.viz = NeuroVizWidget()
        layout.addWidget(self.viz)
        self.setLayout(layout)

    def update_metrics(self, metrics):
        self.viz.update_state(metrics)

    def trigger_kalki_flash(self):
        """Ativa o pulso de luz branca (A Espada)"""
        self.viz.flash_intensity = 1.0
        # Reset geometry to stable golden state
        self.viz.coherence = 1.0
        self.viz.generate_geometry()

class AvalonMainWindow(NeuroVizWindow):
    """Alias for compatibility."""
    def update_display(self, coherence, protocol, time_remaining):
        self.update_metrics({'coherence': coherence})
    def get_state(self):
        return {'fps_stability': 0.95}
