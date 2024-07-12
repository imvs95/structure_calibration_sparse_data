"""
Created on: 31-1-2022 12:09

@author: IvS
"""
class CallBack():
    def __init__(self):
        # Setup any attributes for convergence info
        self.nfe = []
        self.eprogress = []
        #self.hypervolume = []
        #self.hypervolume_func = Hypervolume(minimum=minimum, maximum=maximum)

    def __call__(self, algorithm):
        self.nfe.append(algorithm.algorithm.nfe)
        self.eprogress.append(algorithm.algorithm.archive.improvements)
        #self.hypervolume.append(self.hypervolume_func.calculate(algorithm.algorithm.archive))
