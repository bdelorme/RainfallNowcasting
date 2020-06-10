from pathlib import Path
from subprocess import run


for param_file in Path('experiments').glob('convdlrm*.json'):
    print(f'Run {param_file.stem}')
    run(['python', 'train_model.py', str(param_file)])
    run(['python', 'test_model.py', '--nargs', str(param_file), '10', '2018', '3'])
