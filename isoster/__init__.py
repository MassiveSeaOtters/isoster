"""ISOSTER: accelerated elliptical isophote fitting for galaxy images."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("isoster")
except PackageNotFoundError:  # not installed (e.g. a source checkout on sys.path)
    __version__ = "0.0.0+unknown"

from .config import IsosterConfig
from .driver import fit_image
from .fitting import fit_isophote
from .model import build_ellipse_model, build_isoster_model  # deprecated, removal planned for v0.3
from .plotting import (
    METHOD_STYLES,
    build_method_profile,
    plot_comparison_qa_figure,
    plot_qa_summary,
    plot_qa_summary_extended,
)
from .utils import (
    isophote_results_from_asdf,
    isophote_results_from_fits,
    isophote_results_to_asdf,
    isophote_results_to_astropy_tables,
    isophote_results_to_fits,
)
