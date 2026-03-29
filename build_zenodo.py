#!/usr/bin/env python3
"""Build Zenodo upload ZIP for 15Paper.

Run from /home/rodo/15Paper/:
    python build_zenodo.py

Creates: structural-inductive-bias-v1.0.zip
"""

import os
import zipfile
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_NAME = 'structural-inductive-bias-v1.0.zip'
PREFIX = 'structural-inductive-bias'  # folder name inside ZIP

# Files to include (relative to PROJECT_ROOT)
INCLUDE = {
    'README_zenodo.md': f'{PREFIX}/README.md',
    'LICENSE_zenodo': f'{PREFIX}/LICENSE',
    'requirements.txt': f'{PREFIX}/requirements.txt',
    # Source code
    'src/__init__.py': f'{PREFIX}/src/__init__.py',
    'src/models.py': f'{PREFIX}/src/models.py',
    'src/systems.py': f'{PREFIX}/src/systems.py',
    'src/ham.py': f'{PREFIX}/src/ham.py',
    'src/utils.py': f'{PREFIX}/src/utils.py',
    'src/experiment1.py': f'{PREFIX}/src/experiment1.py',
    'src/experiment2.py': f'{PREFIX}/src/experiment2.py',
    'src/experiment3.py': f'{PREFIX}/src/experiment3.py',
    'src/experiment4.py': f'{PREFIX}/src/experiment4.py',
    'src/experiment5_analytical.py': f'{PREFIX}/src/experiment5_analytical.py',
    'src/barron_analysis.py': f'{PREFIX}/src/barron_analysis.py',
    'src/regenerate_figures.py': f'{PREFIX}/src/regenerate_figures.py',
    # Results (data)
    'results/exp1/experiment1.npz': f'{PREFIX}/results/exp1/experiment1.npz',
    'results/exp2/experiment2.npz': f'{PREFIX}/results/exp2/experiment2.npz',
    'results/exp3/experiment3.npz': f'{PREFIX}/results/exp3/experiment3.npz',
    'results/exp4/experiment4.npz': f'{PREFIX}/results/exp4/experiment4.npz',
    'results/exp5/experiment5.npz': f'{PREFIX}/results/exp5/experiment5.npz',
}


def build():
    zip_path = os.path.join(PROJECT_ROOT, ZIP_NAME)
    
    # Check that required staging files exist, create if needed
    readme_path = os.path.join(PROJECT_ROOT, 'README_zenodo.md')
    license_path = os.path.join(PROJECT_ROOT, 'LICENSE_zenodo')
    
    if not os.path.exists(readme_path):
        print(f"ERROR: {readme_path} not found. Run this after creating staging files.")
        return
    if not os.path.exists(license_path):
        print(f"ERROR: {license_path} not found. Run this after creating staging files.")
        return
    
    print(f"Building {ZIP_NAME}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src_rel, dst_in_zip in INCLUDE.items():
            src_abs = os.path.join(PROJECT_ROOT, src_rel)
            if os.path.exists(src_abs):
                zf.write(src_abs, dst_in_zip)
                size = os.path.getsize(src_abs)
                print(f"  + {dst_in_zip}  ({size:,} bytes)")
            else:
                print(f"  ! MISSING: {src_rel}")

    total = os.path.getsize(zip_path)
    print(f"\nDone: {zip_path}")
    print(f"Total size: {total:,} bytes ({total/1024:.1f} KB)")
    print(f"\nUpload this file to https://zenodo.org/deposit/new")


if __name__ == '__main__':
    build()
