from pathlib import Path
from subprocess import run


#for param_file in Path('experiments').glob('ddnet*.json'):
for param_file in Path('experiments').glob('ddnet_lsm.json'):
    print(f'Run {param_file.stem}')
    run(['python', 'run_one_model.py', str(param_file)])
    run(['python', 'apply_model_to_new_test.py', '--nargs', str(param_file), '13', '2018', '3'])
