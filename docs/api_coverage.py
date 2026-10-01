"""Fail documentation builds that silently omit the public Python API."""

import mhcflurry

# Methods that the guides link to directly.
REQUIRED_METHODS = (
    "mhcflurry.Class1AffinityPredictor.calibrate_percentile_ranks",
    "mhcflurry.Class1ProcessingPredictor.percentile_ranks",
    "mhcflurry.Class1PresentationPredictor.percentile_ranks",
    "mhcflurry.CompactPercentRankTransform.transform",
)

# Every public class, so a new export cannot silently go undocumented.
REQUIRED_API_OBJECTS = tuple(
    "mhcflurry.%s" % name for name in mhcflurry.__all__
    if not name.startswith("__")
) + REQUIRED_METHODS


def check_public_api(app, exception):
    """Require actual Python-domain entries, not just module headings."""
    if exception is not None:
        return
    documented = {entry[0] for entry in app.env.get_domain("py").get_objects()}
    missing = sorted(set(REQUIRED_API_OBJECTS) - documented)
    if missing:
        raise RuntimeError(
            "Public API documentation is missing: %s. Check that generated "
            "RST is parsed as reStructuredText, not Markdown." % ", ".join(missing))


def setup(app):
    app.connect("build-finished", check_public_api)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
