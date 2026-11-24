"""Run the RealWaste-only workflow, stopping immediately on failure."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--realwaste-path',type=Path)
    p.add_argument('--head-epochs',type=int,default=15)
    p.add_argument('--fine-epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=32)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    def run(*arguments):
        print('Running:',sys.executable,*map(str,arguments),flush=True)
        subprocess.run([sys.executable,*map(str,arguments)],cwd=root,check=True)
    acquire=['-m','src.download_dataset']
    if a.realwaste_path: acquire+=['--realwaste-path',a.realwaste_path.resolve()]
    run(*acquire)
    # A supplied extracted folder is not copied by the acquisition helper.
    prepare=['-m','src.prepare_dataset']
    if a.realwaste_path and a.realwaste_path.is_dir(): prepare+=['--realwaste-path',a.realwaste_path.resolve()]
    run(*prepare)
    run('-m','unittest','discover','-s','tests','-v')
    run('-m','src.train','--smoke')
    run('-m','src.train','--head-epochs',a.head_epochs,'--fine-epochs',a.fine_epochs,'--batch',a.batch)
    run('-m','src.verify_export')
    run('-m','src.evaluate')

if __name__=='__main__': main()
