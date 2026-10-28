"""Prepare verified review inputs beside the portable code without changing code paths."""
from pathlib import Path
import csv,hashlib,io,json,shutil,tempfile,zipfile

ROOT=Path(__file__).resolve().parents[2]
NAME='PC_AEGG_Reviewer_Package_20261028.zip'
EXPECTED_SHA256='edf6d8b91ed35f8624dd0c6c44045ca3a060134fdb4fbc1b17f753541adee39e'

def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):sha.update(block)
    return sha.hexdigest()

def prepare():
    archive=next((p for p in (ROOT/NAME,ROOT.parent/NAME) if p.is_file()),None)
    if archive is None:raise FileNotFoundError(f'Place {NAME} in this project directory or its parent directory.')
    if digest(archive)!=EXPECTED_SHA256:raise ValueError('The review archive checksum does not match this input version.')
    count=0
    with zipfile.ZipFile(archive) as z:
        rows=list(csv.DictReader(io.StringIO(z.read('REVIEW_FILE_MANIFEST.csv').decode('utf-8-sig'))))
        for row in rows:
            name=row['relative_path']
            if not name.startswith(('Source_Data/','Micro_CT/')):continue
            target=(ROOT/name).resolve()
            if not target.is_relative_to(ROOT.resolve()):raise ValueError('Unsafe archive member path.')
            if target.exists():
                if target.stat().st_size!=int(row['bytes']) or digest(target)!=row['sha256']:
                    raise FileExistsError(f'Existing input differs: {name}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                handle=tempfile.NamedTemporaryFile(mode='wb',dir=target.parent,prefix=target.name+'.',suffix='.preparing',delete=False)
                temporary=Path(handle.name)
                try:
                    with z.open(name) as source,handle as destination:shutil.copyfileobj(source,destination,4*1024*1024)
                    if temporary.stat().st_size!=int(row['bytes']) or digest(temporary)!=row['sha256']:
                        raise ValueError(f'Input checksum mismatch: {name}')
                    temporary.replace(target)
                finally:
                    if temporary.exists():temporary.unlink()
            count+=1
    assert count>0
    result={'input_archive':NAME,'input_archive_sha256':EXPECTED_SHA256,'files_verified':count,
            'input_snapshot_commit':'206b527387ecc851bb60fcd1e25fb989cad134ac',
            'code_files_replaced':False}
    (ROOT/'REVIEW_INPUT_VERSION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':prepare()
