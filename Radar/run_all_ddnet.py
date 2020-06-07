from pathlib import Path
from subprocess import run


#for param_file in Path('experiments').glob('ddnet*.json'):
for param_file in Path('experiments').glob('test_rapide.json'):
    print(f'Run {param_file.stem}')
    run(['python', 'run_one_model.py', str(param_file)])
