def test_triton_scripts_present():
    assert __import__('pathlib').Path('scripts/triton/start_triton.sh').exists()
    assert __import__('pathlib').Path('scripts/triton/rename_models.sh').exists()

