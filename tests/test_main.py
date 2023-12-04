from openenergyid import __version__
from openenergyid.main import main

def test_version():
    assert __version__ == '0.0.1'

def test_main():
    assert main() == 0
