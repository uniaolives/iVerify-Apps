import numpy as np

class SchmidtBridgeHexagonal:
    def __init__(self, lambdas=None):
        if lambdas is None:
            self.lambdas = np.array([1/6] * 6)
        else:
            self.lambdas = np.array(lambdas)
        self.coherence_factor = self.calculate_coherence()

    def calculate_coherence(self):
        # Simplificação: pureza do estado de Schmidt
        return np.sum(self.lambdas**2)
