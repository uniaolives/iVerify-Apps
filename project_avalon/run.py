#!/usr/bin/env python3
# project_avalon/run.py
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from project_avalon.avalon_core import AvalonCore, main as cli_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        from PyQt5.QtWidgets import QApplication
        from project_avalon.visual.opengl_viz import NeuroVizWindow
        from project_avalon.web.dashboard import AvalonDashboard

        app = QApplication(sys.argv)
        core = AvalonCore()
        core.bootstrap()

        window = NeuroVizWindow()
        core.modules['visual'] = window
        window.show()

        dashboard = AvalonDashboard(core)
        dashboard.start()

        # Start a background session automatically for demo
        import threading
        t = threading.Thread(target=core.start_session, args=('flow', 300))
        t.daemon = True
        t.start()

        sys.exit(app.exec_())
    else:
        cli_main()
