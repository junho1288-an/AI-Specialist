"""Validate generated notebooks without overwriting student code or saving answers."""
import ast
import asyncio
import copy
import hashlib
import json
import re
from pathlib import Path
import nbformat
from nbclient import NotebookClient

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    manifest=json.loads((HERE/'source-manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['sources'].items():
        assert hashlib.sha256((ROOT/'vision'/name).read_bytes()).hexdigest()==digest, f'Source changed: {name}'
    files=['vision-exercises.ipynb','01-resnet-exercises.ipynb','02-vit-exercises.ipynb',
           '03-detr-exercises.ipynb','04-unet-exercises.ipynb']
    counts=[32,8,11,5,8]
    # Validate every split notebook in a clean namespace, problem by problem.
    isolated={}
    for filename,count in zip(files,counts):
        book=nbformat.read(HERE/filename,4);nbformat.validate(book)
        assert (len(book.cells)-2)//4==count
        for index in range(2,len(book.cells),4):
            env={};exec(book.cells[1].source,env)
            answer=re.findall(r'```python\n(.*?)\n```',book.cells[index+3].source,re.S)[0]
            exec(answer,env)
            checker=next(t for t in ast.parse(book.cells[index+2].source).body if isinstance(t,ast.FunctionDef))
            exec(compile(ast.Module(body=[checker],type_ignores=[]),'checker','exec'),env)
            name=checker.name.replace('check_','solve_')
            env[checker.name](env[name])
        isolated[filename]=count
    print('All 4 split notebooks and combined answers passed isolated checks.',flush=True)
    if hasattr(asyncio,'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    kernel={}
    for mode in ['blank','answer']:
        book=nbformat.read(HERE/files[0],4)
        if mode=='answer':
            for index in range(2,len(book.cells),4):
                book.cells[index+1].source=re.findall(r'```python\n(.*?)\n```',book.cells[index+3].source,re.S)[0]
        NotebookClient(book,timeout=120,kernel_name='python3').execute()
        output=''.join(o.get('text','') for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]))
        expected='PASS' if mode=='answer' else 'FAIL'
        opposite='FAIL' if mode=='answer' else 'PASS'
        assert output.count(expected)==32 and output.count(opposite)==0,output
        kernel[mode]=dict(pass_count=output.count('PASS'),fail_count=output.count('FAIL'))
        print(f'Fresh kernel {mode}: {expected} x32',flush=True)
    report=json.loads((HERE/'validation.json').read_text(encoding='utf-8'))
    assert len(report)==32 and all(s['mutation_rejected'] for s in report)
    (HERE/'kernel-validation.json').write_text(json.dumps(dict(isolated=isolated,fresh_kernel=kernel,
        semantic_mutations_rejected=32,source_hashes_unchanged=True),indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
