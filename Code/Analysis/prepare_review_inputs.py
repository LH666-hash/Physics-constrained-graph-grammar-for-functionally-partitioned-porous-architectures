"""Prepare the verified saved-result, plotting and analytic-check inputs."""
from pathlib import Path
from contextlib import ExitStack
import csv,hashlib,io,json,shutil,tempfile,zipfile

ROOT=Path(__file__).resolve().parents[2]
ARCHIVES=[
    ('PC_AEGG_Review_Inputs_20261028.zip', 'f3afe91d5050be0bc5f7c3f9085214c3ada632b500e46a03f9f0e131124eda41', 'REVIEW_FILE_MANIFEST.csv'),
    ('PC_AEGG_Plotting_Inputs_20261029.zip', 'f424e871f7626b914bb366ebcf5ec3b09c33b740844088f1d9283a5d901df0c2', 'PLOTTING_FILE_MANIFEST.csv'),
]


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):sha.update(block)
    return sha.hexdigest()


def prepare():
    combined={};previous_hashes={};records=[]
    with ExitStack() as stack:
        for name,expected,manifest in ARCHIVES:
            path=next((p for p in (ROOT/name,ROOT.parent/name) if p.is_file()),None)
            if path is None:raise FileNotFoundError(f'Place {name} in the project directory or its parent directory.')
            if digest(path)!=expected:raise ValueError(f'Archive checksum mismatch: {name}')
            z=stack.enter_context(zipfile.ZipFile(path))
            rows=list(csv.DictReader(io.StringIO(z.read(manifest).decode('utf-8-sig'))))
            for row in rows:
                member=row['relative_path']
                if not member.startswith(('Source_Data/','Micro_CT/')):continue
                if member in combined:previous_hashes.setdefault(member,set()).add(combined[member][0]['sha256'])
                combined[member]=(row,z)
            records.append({'archive':name,'sha256':expected})
        for name,(row,z) in combined.items():
            target=(ROOT/name).resolve()
            if not target.is_relative_to(ROOT.resolve()):raise ValueError('Unsafe archive member path.')
            if target.exists():
                current=digest(target)
                if target.stat().st_size==int(row['bytes']) and current==row['sha256']:continue
                if current not in previous_hashes.get(name,set()):raise FileExistsError(f'Existing input differs: {name}')
            target.parent.mkdir(parents=True,exist_ok=True)
            handle=tempfile.NamedTemporaryFile(mode='wb',dir=target.parent,prefix=target.name+'.',suffix='.preparing',delete=False)
            temporary=Path(handle.name)
            try:
                with z.open(name) as src,handle as dst:shutil.copyfileobj(src,dst,4*1024*1024)
                if temporary.stat().st_size!=int(row['bytes']) or digest(temporary)!=row['sha256']:raise ValueError(f'Input checksum mismatch: {name}')
                temporary.replace(target)
            finally:
                if temporary.exists():temporary.unlink()
    result={'input_archives':records,'files_verified':len(combined),'code_files_replaced':False}
    (ROOT/'REVIEW_INPUT_VERSION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':prepare()
