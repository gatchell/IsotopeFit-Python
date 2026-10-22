"""Program settings (port of ``config.txt`` / ``settingswindow.m``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass
class Settings:
    """Defaults are those of the MATLAB ``config.txt``.

    ``deltaresolution`` is in percent, ``deltamass`` in mass units;
    ``searchrange`` is in units of the peak sigma.
    """

    minpeakwidth: float = 0.1
    deltaresolution: float = 0.0
    deltamass: float = 0.01
    searchrange: float = 1.0
    fittingmethod_main: str = "linear_system"
    fittingmethod_cal: str = "simplex_lin_combi"

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> Settings:
        """Read a MATLAB-style ``config.txt`` (``name  =  value`` per line)."""
        values: dict[str, object] = {}
        known = {f.name: f.type for f in fields(cls)}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                key, raw = key.strip(), raw.strip()
                if key not in known:
                    continue
                values[key] = raw if known[key] in (str, "str") else float(raw)
        return cls(**values)

    def to_file(self, path: str | os.PathLike) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for f in fields(self):
                v = getattr(self, f.name)
                fh.write(f"{f.name}  =  {v:g}\n" if isinstance(v, float) else f"{f.name}  =  {v}\n")
