import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os


def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
        return True
    except Exception as e:
        print(f"Error executing notebook {notebook_path}: {str(e)}")
        return False


@pytest.mark.parametrize(
    "notebook",
    [
        "demo_capacity_analysis.ipynb",
        "demo_dyntar_analysis.ipynb",
        "demo_energiedelen.ipynb",
        "demo_energyid_download.ipynb",
        "demo_mvlr.ipynb",
        "download_prices.ipynb",
    ],
)
def test_notebooks(notebook):
    notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", notebook))
    assert run_notebook(notebook_path), f"Notebook {notebook} failed to execute"


if __name__ == "__main__":
    pytest.main([__file__])
