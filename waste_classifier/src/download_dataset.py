"""Download official archives or accept local ZIPs/folders; safely extract ZIPs."""
import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile
import json

def check_ucimlrepo():
    """Record whether UCI's tabular client supplies actual image bytes/paths."""
    try:
        from ucimlrepo import fetch_ucirepo
        result = fetch_ucirepo(id=908)
        features = result.data.features
        return {'available': features is not None, 'columns': list(features.columns) if features is not None else [],
                'decision': 'Use the official image archive; tabular metadata is not a verified image collection.'}
    except Exception as exc:
        return {'available': False, 'error': str(exc), 'decision': 'Use official UCI image archive.'}

SOURCES = {
    'realwaste': 'https://archive.ics.uci.edu/static/public/908/realwaste.zip',
    'trashnet': 'https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip',
}

def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace('\\', '/')
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or ':' in name or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f'Unsafe ZIP member: {name}')
        z.extractall(destination)
    return destination

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for source in SOURCES:
        parser.add_argument(f'--{source}-path', type=Path)
    args = parser.parse_args()
    log = Path('outputs/logs'); log.mkdir(parents=True,exist_ok=True)
    (log/'ucimlrepo_check.json').write_text(json.dumps(check_ucimlrepo(),indent=2))
    for source, url in SOURCES.items():
        if source == 'trashnet' and args.trashnet_path is None:
            continue
        local = getattr(args, source + '_path')
        root = Path('data/raw')
        root.mkdir(parents=True, exist_ok=True)
        if local is None:
            local = root / (source + '.zip')
            if not local.exists():
                partial = local.with_suffix('.partial')
                try:
                    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
                        shutil.copyfileobj(response, output)
                    if not zipfile.is_zipfile(partial):
                        raise ValueError('Download is not a ZIP archive')
                    partial.replace(local)
                except Exception as exc:
                    raise SystemExit(f'{source} download failed: {exc}\nDownload {url} manually, then run python -m src.download_dataset --{source}-path PATH_TO_ZIP')
        if local.is_dir():
            print(source, 'local folder:', local.resolve())
        elif local.is_file() and zipfile.is_zipfile(local):
            print(source, safe_extract(local, root / source))
        else:
            raise SystemExit(f'Invalid {source} ZIP/folder: {local}')

if __name__ == '__main__':
    main()
