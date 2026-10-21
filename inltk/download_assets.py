"""Download and verify the model files iNLTK needs for each language.

Downloads are synchronous and use only the standard library, so they work the
same in scripts, Jupyter and Colab (where an event loop is already running).
Every download is checked before it is kept: a failed or HTML response is
never written into the models folder, so a bad download can be retried
instead of leaving a corrupt file behind.
"""
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import LanguageCodes, LMConfigs, AllLanguageConfig

all_language_codes = LanguageCodes()

path = Path(__file__).parent

_RETRIES = 3
_TIMEOUT_SECONDS = 60
_CHUNK = 1 << 16
# Error and interstitial pages from file hosts start like this.
_HTML_MARKERS = (b'<!doctype html', b'<html', b'<?xml', b'<head')


class ModelDownloadError(RuntimeError):
    """Raised when a model file could not be downloaded or is not valid."""


def _looks_like_html(first_bytes: bytes) -> bool:
    return first_bytes.lstrip().lower().startswith(_HTML_MARKERS)


def is_valid_model_file(file_path) -> bool:
    """Return True if file_path exists, is non-empty and is not an HTML page."""
    file_path = Path(file_path)
    try:
        if not file_path.is_file() or file_path.stat().st_size == 0:
            return False
        with open(file_path, 'rb') as f:
            return not _looks_like_html(f.read(512))
    except OSError:
        return False


def _fetch(url: str, tmp_path: Path) -> None:
    request = urllib.request.Request(url, headers={'User-Agent': 'inltk'})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        status = getattr(response, 'status', 200)
        if status != 200:
            raise ModelDownloadError(f'Server returned HTTP {status} for {url}')
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type.lower():
            raise ModelDownloadError(f'Got a web page instead of a model file from {url}. '
                                     f'The download link may have expired.')
        with open(tmp_path, 'wb') as out:
            shutil.copyfileobj(response, out, _CHUNK)
    if not is_valid_model_file(tmp_path):
        raise ModelDownloadError(f'The file downloaded from {url} is empty or is a web page, '
                                 f'not a model file. The download link may have expired.')


def download_file(url: str, dest, fname: str, retries: int = _RETRIES) -> bool:
    """Download url to dest/fname. Returns True if downloaded, False if a valid copy already exists.

    The file is written to a temporary name and only moved into place once it
    has been checked, so an interrupted or failed download never leaves a
    corrupt file that later looks installed.
    """
    dest = Path(dest)
    target = dest / fname
    if is_valid_model_file(target):
        return False
    if target.exists():
        # A previous download left a broken file (for example an HTML error page). Replace it.
        target.unlink()
    os.makedirs(dest, exist_ok=True)
    print(f'Downloading {fname}. This might take time, depending on your internet connection. '
          f'Please be patient.\nWe\'ll only do this for the first time.')
    last_error = None
    for attempt in range(1, retries + 1):
        fd, tmp_name = tempfile.mkstemp(dir=dest, prefix=f'.{fname}.', suffix='.part')
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            _fetch(url, tmp_path)
            os.replace(tmp_path, target)
            return True
        except (urllib.error.URLError, OSError, ModelDownloadError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    raise ModelDownloadError(f'Could not download {fname} after {retries} attempts: {last_error}')


def setup_language_sync(language_code: str) -> bool:
    lmconfig = LMConfigs(language_code)
    config = lmconfig.get_config()
    language_dir = path / 'models' / f'{language_code}'
    download_file(config['lm_model_url'], language_dir, config['lm_model_file_name'])
    download_file(config['tokenizer_model_url'], language_dir, config['tokenizer_model_file_name'])
    print('Done!')
    return True


async def setup_language(language_code: str) -> bool:
    """Kept for backwards compatibility; prefer setup_language_sync."""
    return setup_language_sync(language_code)


def verify_language(language_code: str) -> bool:
    lmconfig = LMConfigs(language_code)
    config = lmconfig.get_config()
    language_dir = path / 'models' / f'{language_code}'
    return is_valid_model_file(language_dir / config['lm_model_file_name']) and \
        is_valid_model_file(language_dir / config['tokenizer_model_file_name'])


def check_all_languages_identifying_model_sync() -> bool:
    config = AllLanguageConfig.get_config()
    all_dir = path / 'models' / 'all'
    downloaded_model = download_file(config['all_languages_identifying_model_url'], all_dir,
                                     config['all_languages_identifying_model_name'])
    downloaded_tokenizer = download_file(config['all_languages_identifying_tokenizer_url'], all_dir,
                                         config['all_languages_identifying_tokenizer_name'])
    return downloaded_model or downloaded_tokenizer


async def check_all_languages_identifying_model() -> bool:
    """Kept for backwards compatibility; prefer check_all_languages_identifying_model_sync."""
    return check_all_languages_identifying_model_sync()
