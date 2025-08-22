from pathlib import Path

def test_examples_presence():
    assert Path('examples/end_to_end_demo.py').exists()
    assert Path('examples/config/config.yaml').exists()

