import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raven_app.process_runner import ProcessRunner


class _FakeProcess:
    stdout = ('stage one\n', 'stage two\n')

    def wait(self):
        return 0


class ProcessRunnerTests(unittest.TestCase):
    def test_streamed_output_and_command_are_saved_in_per_run_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / 'logs' / 'workflow.log'
            runner = ProcessRunner()
            runner._args = ['workflow', '--output', tmp]
            runner._log_path = log_path
            runner._initial_log = '>>> Starting workflow'
            with patch('raven_app.process_runner.subprocess.Popen', return_value=_FakeProcess()) as popen:
                runner.run()

            log = log_path.read_text(encoding='utf-8')
            self.assertIn('Command:', log)
            self.assertIn('>>> Starting workflow', log)
            self.assertLess(log.index('stage one'), log.index('stage two'))
            self.assertIn('(exit code 0)', log)
            flags = popen.call_args.kwargs['creationflags']
            expected = getattr(__import__('subprocess'), 'CREATE_NEW_PROCESS_GROUP', 0) if os.name == 'nt' else 0
            self.assertEqual(flags, expected)
            if os.name == 'nt':
                startupinfo = popen.call_args.kwargs['startupinfo']
                self.assertTrue(startupinfo.dwFlags & __import__('subprocess').STARTF_USESHOWWINDOW)
                self.assertEqual(startupinfo.wShowWindow, __import__('subprocess').SW_HIDE)
            else:
                self.assertIsNone(popen.call_args.kwargs['startupinfo'])


if __name__ == '__main__':
    unittest.main()
