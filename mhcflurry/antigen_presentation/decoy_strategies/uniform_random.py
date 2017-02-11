import numpy

from .decoy_strategy import DecoyStrategy


class UniformRandom(DecoyStrategy):
    """
    Decoy strategy that selects decoys randomly from a provided universe
    of peptides.
    """
    def __init__(self, all_peptides, decoys_per_hit=999):
        DecoyStrategy.__init__(self)
        self.all_peptides = set(all_peptides)
        self.decoys_per_hit = decoys_per_hit

    def decoys_for_experiment(self, experiment_name, hit_list):
        decoy_pool = self.all_peptides.difference(set(hit_list))
        return numpy.random.choice(
            list(decoy_pool),
            replace=True,
            size=self.decoys_per_hit * len(hit_list))
