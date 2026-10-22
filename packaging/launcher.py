"""Entry point of the macOS app bundle (see IsotopeFit.spec)."""

import sys

from isotopefit.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
