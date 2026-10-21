"""Tests for the parts of iNLTK that work without fastai or downloaded models.

Run with:  python -m unittest discover -s tests -v
"""
import asyncio
import http.server
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from inltk import download_assets
from inltk.download_assets import ModelDownloadError, download_file, is_valid_model_file
from inltk.utils import cos_sim

MODEL_BYTES = b'\x0a\x0b binary model payload \x00\x01' * 50
HTML_PAGE = b'<!DOCTYPE html><html><body>This link has expired</body></html>'


class _Handler(http.server.BaseHTTPRequestHandler):
    routes = {
        '/model.bin': (200, 'application/octet-stream', MODEL_BYTES),
        '/expired.bin': (200, 'text/html; charset=utf-8', HTML_PAGE),
        '/sneaky.bin': (200, 'application/octet-stream', HTML_PAGE),  # HTML with a wrong content type
        '/empty.bin': (200, 'application/octet-stream', b''),
    }
    hits = {}

    def do_GET(self):
        _Handler.hits[self.path] = _Handler.hits.get(self.path, 0) + 1
        if self.path not in self.routes:
            self.send_error(404)
            return
        status, content_type, body = self.routes[self.path]
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / 'te'
        _Handler.hits.clear()
        # No real waiting between retries in tests.
        patcher = mock.patch.object(download_assets.time, 'sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def leftover_files(self):
        return sorted(p.name for p in self.dest.iterdir()) if self.dest.exists() else []

    def test_downloads_valid_file(self):
        self.assertTrue(download_file(f'{self.base}/model.bin', self.dest, 'tokenizer.model'))
        self.assertEqual((self.dest / 'tokenizer.model').read_bytes(), MODEL_BYTES)
        self.assertEqual(self.leftover_files(), ['tokenizer.model'])

    def test_html_error_page_is_rejected_and_not_saved(self):
        # Issue #99: a host error page used to be saved as tokenizer.model.
        with self.assertRaises(ModelDownloadError):
            download_file(f'{self.base}/expired.bin', self.dest, 'tokenizer.model', retries=2)
        self.assertEqual(self.leftover_files(), [])

    def test_html_body_with_wrong_content_type_is_rejected(self):
        with self.assertRaises(ModelDownloadError):
            download_file(f'{self.base}/sneaky.bin', self.dest, 'tokenizer.model', retries=1)
        self.assertEqual(self.leftover_files(), [])

    def test_empty_body_is_rejected(self):
        with self.assertRaises(ModelDownloadError):
            download_file(f'{self.base}/empty.bin', self.dest, 'tokenizer.model', retries=1)
        self.assertEqual(self.leftover_files(), [])

    def test_http_error_is_retried_then_raises(self):
        with self.assertRaises(ModelDownloadError):
            download_file(f'{self.base}/missing.bin', self.dest, 'tokenizer.model', retries=3)
        self.assertEqual(_Handler.hits.get('/missing.bin'), 3)
        self.assertEqual(self.leftover_files(), [])

    def test_existing_valid_file_is_not_downloaded_again(self):
        self.dest.mkdir(parents=True)
        (self.dest / 'tokenizer.model').write_bytes(MODEL_BYTES)
        self.assertFalse(download_file(f'{self.base}/model.bin', self.dest, 'tokenizer.model'))
        self.assertNotIn('/model.bin', _Handler.hits)

    def test_existing_broken_file_is_replaced(self):
        # Before the fix, a broken file blocked every future download because it "existed".
        self.dest.mkdir(parents=True)
        (self.dest / 'tokenizer.model').write_bytes(HTML_PAGE)
        self.assertTrue(download_file(f'{self.base}/model.bin', self.dest, 'tokenizer.model'))
        self.assertEqual((self.dest / 'tokenizer.model').read_bytes(), MODEL_BYTES)

    def test_is_valid_model_file(self):
        self.dest.mkdir(parents=True)
        good, html, empty = self.dest / 'a', self.dest / 'b', self.dest / 'c'
        good.write_bytes(MODEL_BYTES)
        html.write_bytes(b'  \n' + HTML_PAGE)
        empty.write_bytes(b'')
        self.assertTrue(is_valid_model_file(good))
        self.assertFalse(is_valid_model_file(html))
        self.assertFalse(is_valid_model_file(empty))
        self.assertFalse(is_valid_model_file(self.dest / 'missing'))


class SetupInNotebookTests(unittest.TestCase):
    def test_setup_works_inside_a_running_event_loop(self):
        # Issue #99: setup() raised "This event loop is already running" in Jupyter and Colab.
        from inltk import inltk as api
        with mock.patch.object(api, 'setup_language_sync', return_value=True) as fake_setup:
            async def notebook_cell():
                return api.setup('te')
            self.assertTrue(asyncio.run(notebook_cell()))
        fake_setup.assert_called_once_with('te')

    def test_setup_rejects_unknown_language(self):
        from inltk import inltk as api
        with self.assertRaises(Exception):
            api.setup('xx')


class OptionalFastaiTests(unittest.TestCase):
    def test_module_imports_without_fastai(self):
        from inltk import inltk as api
        self.assertTrue(callable(api.tokenize))

    def test_model_features_explain_missing_fastai(self):
        from inltk import inltk as api
        if api._FASTAI_IMPORT_ERROR is None:
            self.skipTest('fastai is installed')
        with self.assertRaises(ImportError) as ctx:
            api.predict_next_words('నమస్కారం', 3, 'te')
        self.assertIn('Tokenization works without it', str(ctx.exception))


class CosSimTests(unittest.TestCase):
    def test_identical_orthogonal_and_opposite(self):
        self.assertAlmostEqual(cos_sim([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(cos_sim([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cos_sim([1, 1], [-1, -1]), -1.0)

    def test_zero_vector(self):
        self.assertEqual(cos_sim([0, 0], [1, 2]), 0.0)


if __name__ == '__main__':
    unittest.main()
