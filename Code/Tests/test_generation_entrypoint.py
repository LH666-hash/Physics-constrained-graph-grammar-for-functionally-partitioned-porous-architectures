"""Output reservation regression: launchers may write audits before core.main()."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'Code/Analysis'))
import generate_case


class GenerationEntrypointTests(unittest.TestCase):
    def test_fresh_output_exists_before_launcher_audit(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'new_parent'/'organic_run'

            def launcher(cmd, *, check):
                self.assertTrue(check)
                self.assertTrue(out.is_dir())
                self.assertEqual(list(out.iterdir()),[])
                self.assertEqual(generate_case.initialize_output_directory(
                    out,allow_existing_empty=True),out.resolve())
                (out/'contour_repair_budget.json').write_text(json.dumps({'initialized':True}))

            with patch.object(sys,'argv',['generate_case.py','--case','organic22p5','--out',str(out)]), \
                 patch.object(generate_case.subprocess,'run',side_effect=launcher) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                generate_case.main()
            run.assert_called_once()
            self.assertTrue(json.loads((out/'contour_repair_budget.json').read_text())['initialized'])

    def test_existing_outputs_and_archive_are_protected(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'existing'
            out.mkdir()
            with self.assertRaises(FileExistsError):
                generate_case.initialize_output_directory(out)
            marker=out/'archived_result.json'
            marker.write_bytes(b'original archived bytes\n')
            with patch.object(sys,'argv',['generate_case.py','--case','organic22p5','--out',str(out)]), \
                 patch.object(generate_case.subprocess,'run') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):generate_case.main()
            run.assert_not_called()
            with self.assertRaises(FileExistsError):
                generate_case.initialize_output_directory(out,allow_existing_empty=True)
            self.assertEqual(marker.read_bytes(),b'original archived bytes\n')
        with self.assertRaises(ValueError):
            generate_case.initialize_output_directory(PACKAGE/'Source_Data'/'new_generation_forbidden')

    def test_direct_organic_launcher_rejects_existing_results_before_core(self):
        launcher=generate_case.command('organic22p5',Path('unused'))[0][1]
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'existing'
            out.mkdir()
            marker=out/'archived_result.json'
            marker.write_bytes(b'keep unchanged\n')
            result=subprocess.run([sys.executable,launcher,'--out-dir',str(out)],
                                  capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('FileExistsError',result.stderr)
            self.assertEqual(list(out.iterdir()),[marker])
            self.assertEqual(marker.read_bytes(),b'keep unchanged\n')
        result=subprocess.run([sys.executable,launcher,'--out-dir',str(PACKAGE/'Source_Data')],
                              capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('outside archived Source_Data',result.stderr)


if __name__=='__main__':unittest.main()
