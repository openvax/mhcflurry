"""Verify a built distribution image with network access disabled."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
from mhcflurry import __version__
from mhcflurry.downloads import get_downloads_dir

assert __version__ == sys.argv[1], (__version__, sys.argv[1])
# The default user must be able to refresh current weights and add older releases.
cache = Path(get_downloads_dir())
for directory in (cache, cache.parent):
    with TemporaryDirectory(prefix=".smoke-", dir=directory) as scratch:
        Path(scratch, "writable").write_text("ok")
predictor = Class1PresentationPredictor.load()
peptides = ["TPVCPNGPG", "RLLEGMEMI"]
alleles = ["HLA-A*02:01"]
for flanks in ({"n_flanks": ["MSSSS", "MVENK"], "c_flanks": ["NCQV", "FGQVI"]}, {}):
    result = predictor.predict(peptides, alleles, **flanks)
    assert len(result) == len(peptides)
    assert numpy.isfinite(result[["affinity", "processing_score", "presentation_score"]]).all().all()
    assert result.presentation_score.between(0, 1).all()
    print(result.to_csv(index=False))
numpy.testing.assert_allclose(
    Class1AffinityPredictor.load().predict(peptides, allele=alleles[0]),
    predictor.affinity_predictor.predict(peptides, allele=alleles[0]),
    rtol=1e-6,
)
print("Offline image smoke passed:", __version__)
