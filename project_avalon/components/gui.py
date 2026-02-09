# project_avalon/components/gui.py
import sys
from project_avalon.components.realtime_visualizer import AvalonMainWindow, QApplication

class AvalonGUI:
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.app = QApplication(sys.argv)
        # Use the OpenGL implementation
        self.main_window = AvalonMainWindow(eeg_source=kernel)

    def run(self):
        self.main_window.show()
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    from project_avalon.avalon_kernel import AvalonKernel
    kernel = AvalonKernel()
    gui = AvalonGUI(kernel)
    gui.run()
