from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from orchestrator.trace_retention import (
    collect_evidence_files,
    write_trace_retention_manifest,
)
from orchestrator.campaign import Campaign
from long_horizon.store import CampaignStore


class TraceRetentionManifestTest(unittest.TestCase):
    def workspace(self, root: Path) -> Path:
        workspace = root / "kernel_opt_demo_triton_l20"
        (workspace / "memory").mkdir(parents=True)
        (workspace / ".atrex_long_horizon/episodes/e0001/episode_runtime").mkdir(
            parents=True
        )
        (workspace / ".gpu_wiki_profile/raw/query_events/2026-09-04").mkdir(
            parents=True
        )
        (workspace / "profiles/v1/analysis").mkdir(parents=True)
        (workspace / "profiles/v1/att").mkdir(parents=True)
        (workspace / ".claude").mkdir()
        (workspace / "kernel.py").write_text("def run(): return 1\n")
        (workspace / "helper.cu").write_text("// candidate source\n")
        (workspace / "solution.json").write_text(json.dumps({
            "spec": {"target_hardware": ["L20"]},
            "sources": [{"path": "kernel.py"}, {"path": "helper.cu"}],
        }))
        (workspace / "memory/v0.json").write_text('{"version":0}\n')
        (workspace / "memory/v0_debug.json").write_text('{"debug":true}\n')
        (workspace / "memory/long_horizon_e0001.json").write_text(
            '{"episode":1}\n'
        )
        (workspace / ".atrex_long_horizon/state.json").write_text('{}\n')
        (workspace / ".atrex_long_horizon/evaluations.jsonl").write_text(
            '{"gate":"PASS"}\n'
        )
        episode = workspace / ".atrex_long_horizon/episodes/e0001/episode_runtime"
        (episode / "journal.json").write_text('{"experiments":[]}\n')
        (episode / "evaluations.jsonl").write_text('{"correct":true}\n')
        (workspace / ".gpu_wiki_profile/run.json").write_text('{"run_id":"r"}\n')
        query = workspace / ".gpu_wiki_profile/raw/query_events/2026-09-04/q.json"
        query.write_text('{"event_type":"wiki_query"}\n')
        (workspace / "profiles/v1/REPORT.md").write_text("# compact\n")
        (workspace / "profiles/v1/analysis/metrics_key_run.json").write_text('{}\n')
        (workspace / "profiles/v1/att/raw.att").write_bytes(b"large raw capture")
        (workspace / ".claude/session.jsonl").write_text('{"private":"chat"}\n')
        return workspace

    def test_manifest_is_minimal_and_wiki_mining_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(Path(tmp))
            manifest_path = write_trace_retention_manifest(
                workspace,
                "completed",
                hardware={
                    "platform": "B300",
                    "arch": "sm_103",
                    "sandbox_hardware": "L20D",
                },
            )
            self.assertIsNotNone(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            paths = {row["path"] for row in manifest["files"]}
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["hardware"], {
                "platform": "B300",
                "arch": "sm_103",
                "sandbox_hardware": "L20D",
            })
            for required in (
                "kernel.py",
                "helper.cu",
                "solution.json",
                "memory/v0.json",
                "memory/long_horizon_e0001.json",
                ".atrex_long_horizon/state.json",
                ".atrex_long_horizon/evaluations.jsonl",
                ".atrex_long_horizon/episodes/e0001/episode_runtime/journal.json",
                ".gpu_wiki_profile/run.json",
                ".gpu_wiki_profile/raw/query_events/2026-09-04/q.json",
                "profiles/v1/REPORT.md",
                "profiles/v1/analysis/metrics_key_run.json",
            ):
                self.assertIn(required, paths)
            self.assertNotIn("memory/v0_debug.json", paths)
            self.assertNotIn("profiles/v1/att/raw.att", paths)
            self.assertNotIn(".claude/session.jsonl", paths)

    def test_collection_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(Path(tmp))
            self.assertEqual(
                collect_evidence_files(workspace),
                collect_evidence_files(workspace),
            )

    def test_campaign_routes_query_events_to_ignored_incumbent_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = Campaign(
                name="demo",
                kernel_demo="reference.py",
                platform="L20",
                framework="Triton",
                work_dir=str(root),
                workspace_suffix="triton_l20",
            )
            campaign.workspace.mkdir(parents=True)
            subprocess.run(["git", "init", "-q", str(campaign.workspace)], check=True)
            CampaignStore.ensure_excluded(campaign.workspace)
            environment = campaign.agent_environment(episode_mode="fast")
            self.assertEqual(
                environment["ATREX_WIKI_PROFILE_ROOT"],
                str((campaign.workspace / ".gpu_wiki_profile").resolve()),
            )
            self.assertEqual(environment["ATREX_WIKI_TASK_ID"], campaign.campaign_name)
            exclude = subprocess.run(
                ["git", "-C", str(campaign.workspace), "rev-parse", "--git-path", "info/exclude"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            exclude_path = Path(exclude)
            if not exclude_path.is_absolute():
                exclude_path = campaign.workspace / exclude_path
            text = exclude_path.read_text()
            self.assertIn("/.gpu_wiki_profile/", text)
            self.assertIn("/trace-retention-manifest.json", text)


if __name__ == "__main__":
    unittest.main()
