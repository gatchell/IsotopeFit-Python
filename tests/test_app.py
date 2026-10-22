import sys

from isotopefit.gui.app import use_native_menus


def test_native_menus_only_in_the_mac_app(monkeypatch):
    monkeypatch.delenv("ISOTOPEFIT_NATIVE_MENUS", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert not use_native_menus()  # started from a terminal
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert use_native_menus() == (sys.platform == "darwin")
    monkeypatch.setenv("ISOTOPEFIT_NATIVE_MENUS", "0")
    assert not use_native_menus()
    monkeypatch.setenv("ISOTOPEFIT_NATIVE_MENUS", "1")
    assert use_native_menus()


def test_icon_is_packaged():
    from importlib import resources

    assert (resources.files("isotopefit.gui") / "icon.png").is_file()
