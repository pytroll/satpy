"""EWA resamplers."""

from pyresample.ewa import DaskEWAResampler, LegacyDaskEWAResampler


def get_resampler_classes():
    """Get bucket resampler classes."""
    return {
        "ewa": DaskEWAResampler,
        "ewa_legacy": LegacyDaskEWAResampler,
    }
