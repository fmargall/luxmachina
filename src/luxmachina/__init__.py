from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("luxmachina")
except PackageNotFoundError:
    __version__ = "?.?.?"