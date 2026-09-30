# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Create a reproducible source snapshot of the native runtime and its tests.

Uses the working tree deliberately, including new untracked implementation files.
No git installation, network, package installation or binary redistribution.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
FOLDERS=('core','frontend','runtime','third_party','app','python','tests','examples/grammar')
SUFFIXES={'.c','.h','.py','.cmake','.mk','.json','.md','.exports','.toml','.txt','.csv'}
NAMES={'Makefile','CMakeLists.txt'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();files={}
    for name in ('Makefile','CMakeLists.txt','pyproject.toml','README.md','AGENTS.md'):
        files[name]=(ROOT/name).read_bytes()
    for folder in FOLDERS:
        for path in sorted((ROOT/folder).rglob('*')):
            relative=path.relative_to(ROOT)
            if path.is_symlink() or not path.is_file() or any(x.startswith('.') or x in ('__pycache__','build') for x in relative.parts):continue
            if path.suffix in SUFFIXES or path.name in NAMES:files[str(relative)]=path.read_bytes()
    for path in sorted((ROOT/'docs/grammar').glob('*.md')):
        files[str(path.relative_to(ROOT))]=path.read_bytes()
    manifest=dict(format='odezza.native-source-snapshot.v1',
                  source='working tree; not a tagged release',
                  files={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())})
    files['SOURCE_MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as archive:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo('odezza-native-trial/'+name);info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
    print(json.dumps(dict(path=str(output.resolve()),files=len(files),bytes=output.stat().st_size,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
