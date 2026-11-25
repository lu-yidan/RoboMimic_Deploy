"""Check ACP motor-mode selection with a fake tmux; no DDS or robot access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ACPTmuxTest(unittest.TestCase):
    def run_layout(self, read_only, *, existing=False, stored_acp="1", stored_read_only="1"):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            fake = directory / "tmux"
            fake.write_text("#!" + sys.executable + "\n" + '''
import json, os, sys
args = sys.argv[1:]
with open(os.environ['TMUX_TEST_LOG'], 'a') as output:
    output.write(json.dumps(args) + '\\n')
if args[0] == 'has-session':
    sys.exit(0 if os.environ['TMUX_TEST_EXISTS'] == '1' else 1)
if args[0] == 'show-option':
    print(os.environ['TMUX_TEST_ACP' if args[-1] == '@acp_dlio' else 'TMUX_TEST_READ_ONLY'])
elif args[0] in ('display-message', 'split-window'):
    print('%1')
''')
            fake.chmod(0o755)
            log = directory / "calls.jsonl"
            env = dict(os.environ, PATH=str(directory) + os.pathsep + os.environ["PATH"],
                       TMUX_TEST_LOG=str(log), TMUX_TEST_EXISTS=str(int(existing)),
                       TMUX_TEST_ACP=stored_acp, TMUX_TEST_READ_ONLY=stored_read_only,
                       ACP_DLIO="1", REPO_DIR="/tmp/repo space",
                       PYTHON_BIN="/tmp/python env/bin/python")
            env.pop("TMUX", None)
            env.pop("LYD_DEPLOY_READ_ONLY", None)
            if read_only is not None:
                env["LYD_DEPLOY_READ_ONLY"] = read_only
            result = subprocess.run(
                ["bash", str(ROOT / "tools/deploy/start_tmux_layout.sh"), "test-acp", "--detach"],
                env=env, capture_output=True, text=True,
            )
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return result, calls

    def test_default_prefills_both_sides_read_only(self):
        self.assert_prefilled_mode(None, "1")

    def test_explicit_live_prefills_both_sides_writable(self):
        self.assert_prefilled_mode("0", "0")

    def assert_prefilled_mode(self, requested, expected):
        result, calls = self.run_layout(requested)
        self.assertEqual(result.returncode, 0, result.stderr)
        keys = [call for call in calls if call[0] == "send-keys"]
        self.assertEqual(len(keys), 7)
        self.assertTrue(all("-l" in call and "Enter" not in call and "C-m" not in call for call in keys))
        commands = [call[-1] for call in keys]
        cpp = next(command for command in commands if "cpp_bridge_main" in command)
        policy = next(command for command in commands if "deploy_policy.py" in command)
        self.assertIn("BRIDGE_READ_ONLY=" + expected, cpp)
        self.assertEqual("--read-only" in policy, expected == "1")
        self.assertTrue(any("deploy_acp_dlio.yaml" in command for command in commands))
        self.assertIn(["set-option", "-t", "test-acp", "@deploy_read_only", expected], calls)
        self.assertIn(["set-option", "-t", "test-acp", "@acp_dlio", "1"], calls)

    def test_existing_read_only_cannot_silently_become_live(self):
        self.assert_mode_rejected("0", stored_read_only="1")

    def test_existing_live_cannot_be_misreported_as_read_only(self):
        self.assert_mode_rejected("1", stored_read_only="0")

    def test_legacy_unknown_mode_requires_explicit_inspection(self):
        self.assert_mode_rejected("0", stored_acp="", stored_read_only="")

    def assert_mode_rejected(self, requested, **stored):
        result, calls = self.run_layout(requested, existing=True, **stored)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Existing session", result.stderr)
        self.assertFalse(any(call[0] in ("new-session", "send-keys", "attach-session", "switch-client")
                             for call in calls))

    def test_matching_session_can_be_reused(self):
        result, calls = self.run_layout("0", existing=True, stored_read_only="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(call[0] in ("new-session", "send-keys") for call in calls))


if __name__ == "__main__":
    unittest.main()
