"""Tests for the created_artifacts verification gate added in #25288.

The bug: agents marked Kanban tasks as DONE while the underlying cron
was never created (or was created multiple times because the agent
retried after a misleading earlier error).  These tests pin the new
contract on every layer the fix touched so a future refactor cannot
regress the flow.

Coverage:

* ``cron.jobs.create_job`` idempotency_key — first call mints a job,
  subsequent calls with the same key return the existing job, repeats
  with no key fall through to the legacy "always create" behaviour.
* ``cron.jobs.find_jobs`` — exact-match on idempotency_key,
  case-insensitive substring on name, AND semantics.
* ``kanban_db._normalize_artifacts`` — accepts list-of-dicts AND
  list-of-strings (cron-id shortcut), rejects entries missing ``id``.
* ``kanban_db._verify_created_artifacts`` truth table — verified /
  phantom / advisory bucketing for known and unknown kinds.
* ``kanban_db.complete_task`` end-to-end — phantom cron id blocks
  completion with ``HallucinatedArtifactsError`` and writes the
  ``completion_blocked_artifact_hallucination`` audit event; verified
  artifacts land on the ``completed`` event payload; the task itself
  is not mutated on rejection.
* ``kanban_complete`` tool — phantom artifact returns the structured
  retry-friendly tool_error from #22923 (still in-flight, retry hint),
  retry with corrected list lands the completion, the issue #25288
  failure mode is reproduced and pinned shut.
* ``cronjob`` tool create branch — idempotency_key flows through;
  response.reused=true on the dedup path; reused=false on first call.
* ``KANBAN_GUIDANCE`` prompt — mentions ``created_artifacts``,
  ``idempotency_key``, and the issue number so a regression of the
  prompt teaching catches a unit test rather than the next agent run.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Shared worker_env fixture (mirrors tests/tools/test_kanban_tools.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Isolated $HERMES_HOME, a real Kanban DB, and a single claimed task.

    Kept identical to ``tests/tools/test_kanban_tools.py::worker_env`` so
    the two suites can share testing patterns.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="worker-test", assignee="test-worker",
        )
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory.

    Same shape as the fixture in tests/cron/test_jobs.py — duplicated
    here rather than imported so this suite can run standalone without
    pulling the cron test conftest.
    """
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# ---------------------------------------------------------------------------
# cron.jobs idempotency + find_jobs
# ---------------------------------------------------------------------------

class TestCreateJobIdempotency:
    def test_no_key_creates_distinct_jobs(self, tmp_cron_dir):
        """The pre-existing 'always create' behaviour is preserved when
        no key is passed — important so existing call sites do not
        suddenly start deduping each other."""
        from cron.jobs import create_job

        a = create_job(prompt="ping", schedule="every 30m")
        b = create_job(prompt="ping", schedule="every 30m")
        assert a["id"] != b["id"]

    def test_same_key_returns_existing_job_unchanged(self, tmp_cron_dir):
        from cron.jobs import create_job, list_jobs

        first = create_job(
            prompt="check h13b", schedule="every 30m",
            idempotency_key="kanban:t_25288",
        )
        second = create_job(
            prompt="check h13b", schedule="every 30m",
            idempotency_key="kanban:t_25288",
        )
        assert second["id"] == first["id"]
        assert second["idempotency_key"] == "kanban:t_25288"
        # Storage was not duplicated.
        assert len(list_jobs(include_disabled=True)) == 1

    def test_idempotency_dedup_runs_before_validation(self, tmp_cron_dir, tmp_path):
        """A retry whose underlying script has since been deleted must
        still resolve to the prior job — the goal is "no duplicates",
        not "re-validate every retry".  Mirrors the failure mode where
        an agent's first attempt landed but the second got an
        ambiguous tool_error and retried."""
        from cron.jobs import create_job

        script = tmp_path / "watchdog.sh"
        script.write_text("#!/bin/bash\necho hello\n")
        script.chmod(0o755)

        first = create_job(
            prompt="", schedule="every 1h", script=str(script),
            no_agent=True, idempotency_key="watchdog-1",
        )

        script.unlink()  # delete underlying script

        second = create_job(
            prompt="", schedule="every 1h",
            script=str(script),  # path no longer exists
            no_agent=True, idempotency_key="watchdog-1",
        )
        assert second["id"] == first["id"]

    def test_empty_or_whitespace_key_is_treated_as_absent(self, tmp_cron_dir):
        from cron.jobs import create_job

        a = create_job(prompt="x", schedule="30m", idempotency_key="")
        b = create_job(prompt="x", schedule="30m", idempotency_key="   ")
        assert a["id"] != b["id"]
        assert a.get("idempotency_key") is None
        assert b.get("idempotency_key") is None


class TestFindJobs:
    def test_finds_by_idempotency_key_exact(self, tmp_cron_dir):
        from cron.jobs import create_job, find_jobs

        target = create_job(
            prompt="alpha", schedule="30m", idempotency_key="abc-123",
        )
        create_job(prompt="bravo", schedule="30m", idempotency_key="other")

        matches = find_jobs(idempotency_key="abc-123")
        assert len(matches) == 1
        assert matches[0]["id"] == target["id"]

    def test_finds_by_name_case_insensitive_substring(self, tmp_cron_dir):
        from cron.jobs import create_job, find_jobs

        a = create_job(prompt="x", schedule="30m", name="Watchdog Memory")
        create_job(prompt="x", schedule="30m", name="Daily Briefing")

        matches = find_jobs(name="memory")
        assert len(matches) == 1
        assert matches[0]["id"] == a["id"]

    def test_filters_compose_with_and(self, tmp_cron_dir):
        from cron.jobs import create_job, find_jobs

        create_job(
            prompt="x", schedule="30m", name="Watchdog A",
            idempotency_key="k-1",
        )
        create_job(
            prompt="x", schedule="30m", name="Watchdog B",
            idempotency_key="k-2",
        )
        # Both filters must match.
        assert find_jobs(idempotency_key="k-1", name="watchdog a")
        assert not find_jobs(idempotency_key="k-1", name="watchdog b")

    def test_returns_empty_for_unknown_key(self, tmp_cron_dir):
        from cron.jobs import find_jobs

        assert find_jobs(idempotency_key="never-existed") == []


# ---------------------------------------------------------------------------
# kanban_db._normalize_artifacts + _verify_created_artifacts
# ---------------------------------------------------------------------------

class TestNormalizeArtifacts:
    def test_dict_form_is_passed_through_with_lowered_kind(self):
        from hermes_cli.kanban_db import _normalize_artifacts

        out = _normalize_artifacts(
            [{"kind": "Cron", "id": "abc123def456", "name": "watchdog"}],
        )
        assert out == [{"kind": "cron", "id": "abc123def456", "name": "watchdog"}]

    def test_string_form_is_interpreted_as_cron_id(self):
        from hermes_cli.kanban_db import _normalize_artifacts

        out = _normalize_artifacts(["abc123def456"])
        assert out == [{"kind": "cron", "id": "abc123def456"}]

    def test_missing_id_raises(self):
        from hermes_cli.kanban_db import _normalize_artifacts

        with pytest.raises(ValueError, match="missing required 'id' field"):
            _normalize_artifacts([{"kind": "cron"}])

    def test_blank_string_is_skipped(self):
        from hermes_cli.kanban_db import _normalize_artifacts

        out = _normalize_artifacts(["", "   ", "abc123"])
        assert out == [{"kind": "cron", "id": "abc123"}]

    def test_unknown_top_level_type_raises(self):
        from hermes_cli.kanban_db import _normalize_artifacts

        with pytest.raises(ValueError, match="must be dicts or strings"):
            _normalize_artifacts([42])  # type: ignore[list-item]

    def test_none_returns_empty(self):
        from hermes_cli.kanban_db import _normalize_artifacts

        assert _normalize_artifacts(None) == []


class TestVerifyCreatedArtifacts:
    def test_known_kind_with_existing_id_verified(self, tmp_cron_dir):
        from cron.jobs import create_job
        from hermes_cli.kanban_db import _verify_created_artifacts

        job = create_job(prompt="x", schedule="30m")
        verified, phantom, advisory = _verify_created_artifacts(
            [{"kind": "cron", "id": job["id"]}],
        )
        assert verified == [{"kind": "cron", "id": job["id"]}]
        assert phantom == [] and advisory == []

    def test_known_kind_with_phantom_id_is_phantom(self, tmp_cron_dir):
        from hermes_cli.kanban_db import _verify_created_artifacts

        verified, phantom, advisory = _verify_created_artifacts(
            [{"kind": "cron", "id": "no-such-job"}],
        )
        assert verified == [] and advisory == []
        assert len(phantom) == 1
        assert phantom[0]["kind"] == "cron"
        assert phantom[0]["id"] == "no-such-job"
        assert "no cron job" in (phantom[0].get("reason") or "").lower()

    def test_unknown_kind_lands_on_advisory_bucket(self, tmp_cron_dir):
        """Unknown kinds keep the gate forward-compatible — a plugin
        that ships a new kind ahead of its verifier should not crash
        completions."""
        from hermes_cli.kanban_db import _verify_created_artifacts

        verified, phantom, advisory = _verify_created_artifacts(
            [{"kind": "future_kind", "id": "x1"}],
        )
        assert verified == [] and phantom == []
        assert advisory == [
            {"kind": "future_kind", "id": "x1",
             "reason": "no verifier registered"},
        ]


# ---------------------------------------------------------------------------
# complete_task end-to-end via the kernel
# ---------------------------------------------------------------------------

class TestCompleteTaskArtifactGate:
    def test_phantom_cron_blocks_completion(self, worker_env, tmp_cron_dir):
        from hermes_cli import kanban_db as kb

        conn = kb.connect()
        try:
            with pytest.raises(kb.HallucinatedArtifactsError) as exc_info:
                kb.complete_task(
                    conn, worker_env,
                    summary="created the cron",
                    created_artifacts=[{"kind": "cron", "id": "ghost123"}],
                )
            assert exc_info.value.phantom[0]["id"] == "ghost123"
            # State unchanged — task still in-flight.
            assert kb.get_task(conn, worker_env).status == "running"
        finally:
            conn.close()

    def test_phantom_writes_audit_event(self, worker_env, tmp_cron_dir):
        from hermes_cli import kanban_db as kb

        conn = kb.connect()
        try:
            with pytest.raises(kb.HallucinatedArtifactsError):
                kb.complete_task(
                    conn, worker_env,
                    summary="oops",
                    created_artifacts=[{"kind": "cron", "id": "ghost"}],
                )
            events = kb.list_events(conn, worker_env)
            kinds = [e.kind for e in events]
            assert "completion_blocked_artifact_hallucination" in kinds
            blocked = next(
                e for e in events
                if e.kind == "completion_blocked_artifact_hallucination"
            )
            assert blocked.payload["phantom_artifacts"][0]["id"] == "ghost"
        finally:
            conn.close()

    def test_verified_cron_completes_and_lands_on_completed_event(
        self, worker_env, tmp_cron_dir,
    ):
        from cron.jobs import create_job
        from hermes_cli import kanban_db as kb

        job = create_job(prompt="watchdog", schedule="every 30m")

        conn = kb.connect()
        try:
            ok = kb.complete_task(
                conn, worker_env,
                summary="created cron watchdog",
                created_artifacts=[{"kind": "cron", "id": job["id"]}],
            )
            assert ok is True
            assert kb.get_task(conn, worker_env).status == "done"

            events = kb.list_events(conn, worker_env)
            completed = next(e for e in events if e.kind == "completed")
            assert completed.payload["verified_artifacts"][0]["id"] == job["id"]
        finally:
            conn.close()

    def test_bare_string_artifact_is_interpreted_as_cron_id(
        self, worker_env, tmp_cron_dir,
    ):
        """Convenience path for agents that paste-drop a raw job id."""
        from cron.jobs import create_job
        from hermes_cli import kanban_db as kb

        job = create_job(prompt="x", schedule="30m")

        conn = kb.connect()
        try:
            ok = kb.complete_task(
                conn, worker_env,
                summary="created cron",
                created_artifacts=[job["id"]],
            )
            assert ok is True
        finally:
            conn.close()

    def test_unknown_kind_passes_with_advisory(
        self, worker_env, tmp_cron_dir,
    ):
        """Unknown kinds must NOT block completion — they land on the
        advisory bucket so a plugin can ship a new kind ahead of its
        verifier without breaking completions."""
        from hermes_cli import kanban_db as kb

        conn = kb.connect()
        try:
            ok = kb.complete_task(
                conn, worker_env,
                summary="shipped a thing",
                created_artifacts=[{"kind": "future_kind", "id": "x1"}],
            )
            assert ok is True
            events = kb.list_events(conn, worker_env)
            completed = next(e for e in events if e.kind == "completed")
            assert completed.payload["advisory_artifacts"][0]["id"] == "x1"
        finally:
            conn.close()

    def test_no_artifacts_field_preserves_legacy_completion_path(
        self, worker_env, tmp_cron_dir,
    ):
        """No created_artifacts ⇒ no gate touched ⇒ legacy behaviour."""
        from hermes_cli import kanban_db as kb

        conn = kb.connect()
        try:
            ok = kb.complete_task(conn, worker_env, summary="just done")
            assert ok is True
            events = kb.list_events(conn, worker_env)
            completed = next(e for e in events if e.kind == "completed")
            assert "verified_artifacts" not in completed.payload
            assert "advisory_artifacts" not in completed.payload
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# kanban_complete tool — the path the agent actually hits
# ---------------------------------------------------------------------------

class TestKanbanCompleteToolArtifactGate:
    def test_phantom_cron_returns_retry_friendly_error(
        self, worker_env, tmp_cron_dir,
    ):
        """Reproduces the exact regression from issue #25288 at the
        tool layer: agent claims it created a cron, kernel rejects.

        The error shape MUST match the #22923 contract (still in-flight,
        retry hint, decision tree) so the model does not panic-block
        or crash the run.
        """
        from tools import kanban_tools as kt
        from hermes_cli import kanban_db as kb

        out = kt._handle_complete({
            "summary": "Created cron job ghost123 to monitor h13b",
            "created_artifacts": [{"kind": "cron", "id": "ghost123"}],
        })
        err = json.loads(out).get("error", "")
        assert err, f"expected an error, got {out!r}"
        # Phantom id surfaced verbatim with kind.
        assert "cron=ghost123" in err
        # The retry-is-supported phrasing — same cues a worker reads
        # to decide whether to retry vs block/abandon.  See #22923 for
        # why this matters.
        assert "still in-flight" in err
        # The three-option decision tree (create-and-retry, drop, block).
        assert "actually create the missing artifact" in err
        assert "kanban_block" in err
        # Issue number named so a model that skims the error catches
        # the warning without reading the surrounding guidance.
        assert "#25288" in err

        conn = kb.connect()
        try:
            assert kb.get_task(conn, worker_env).status == "running"
        finally:
            conn.close()

    def test_retry_with_real_cron_id_completes_task(
        self, worker_env, tmp_cron_dir,
    ):
        """The full reproduction: first attempt with phantom → rejected;
        agent actually creates the cron; second attempt with real id →
        task transitions to done.  This is the workflow the new gate
        is designed to enforce."""
        from cron.jobs import create_job
        from hermes_cli import kanban_db as kb
        from tools import kanban_tools as kt

        # First attempt: phantom claim, gate rejects.
        rejected = json.loads(kt._handle_complete({
            "summary": "Created cron",
            "created_artifacts": [{"kind": "cron", "id": "ghost"}],
        }))
        assert rejected.get("error")

        # Worker actually creates the cron now.
        job = create_job(prompt="real watchdog", schedule="every 30m")

        # Second attempt with the real id lands.
        ok = json.loads(kt._handle_complete({
            "summary": f"Created cron {job['id']} to monitor h13b",
            "created_artifacts": [{"kind": "cron", "id": job["id"]}],
        }))
        assert ok.get("ok") is True
        conn = kb.connect()
        try:
            assert kb.get_task(conn, worker_env).status == "done"
        finally:
            conn.close()

    def test_bare_string_form_works_through_the_tool(
        self, worker_env, tmp_cron_dir,
    ):
        from cron.jobs import create_job
        from tools import kanban_tools as kt

        job = create_job(prompt="x", schedule="30m")
        ok = json.loads(kt._handle_complete({
            "summary": "done",
            "created_artifacts": job["id"],  # also accepts a bare string
        }))
        assert ok.get("ok") is True

    def test_invalid_artifact_shape_returns_friendly_error(self, worker_env):
        from tools import kanban_tools as kt

        out = json.loads(kt._handle_complete({
            "summary": "done",
            "created_artifacts": 12345,  # not a list / dict / string
        }))
        assert out.get("error", "").startswith("created_artifacts must be a list")


# ---------------------------------------------------------------------------
# cronjob tool idempotency_key plumbing
# ---------------------------------------------------------------------------

class TestCronjobToolIdempotencyKey:
    def test_first_create_reports_reused_false(
        self, monkeypatch, tmp_path, tmp_cron_dir,
    ):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        out = json.loads(cronjob(
            action="create",
            prompt="check h13b",
            schedule="every 30m",
            idempotency_key="kanban:t_25288",
        ))
        assert out["success"] is True
        assert out["reused"] is False
        assert out["idempotency_key"] == "kanban:t_25288"
        assert out["message"].endswith("created.")

    def test_repeat_create_reports_reused_true_with_same_id(
        self, monkeypatch, tmp_path, tmp_cron_dir,
    ):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        first = json.loads(cronjob(
            action="create",
            prompt="check h13b",
            schedule="every 30m",
            idempotency_key="kanban:t_25288",
        ))
        second = json.loads(cronjob(
            action="create",
            prompt="check h13b",
            schedule="every 30m",
            idempotency_key="kanban:t_25288",
        ))
        assert second["job_id"] == first["job_id"]
        assert second["reused"] is True
        assert "reused" in second["message"]

    def test_no_key_is_non_idempotent(
        self, monkeypatch, tmp_path, tmp_cron_dir,
    ):
        """The default-off behaviour preserves the v1 contract."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        a = json.loads(cronjob(
            action="create", prompt="x", schedule="30m",
        ))
        b = json.loads(cronjob(
            action="create", prompt="x", schedule="30m",
        ))
        assert a["job_id"] != b["job_id"]
        assert a["reused"] is False
        assert b["reused"] is False


# ---------------------------------------------------------------------------
# KANBAN_GUIDANCE prompt teaching
# ---------------------------------------------------------------------------

class TestKanbanGuidanceMentionsArtifactGate:
    def test_guidance_names_created_artifacts_and_cron_verifier(self):
        from agent.prompt_builder import KANBAN_GUIDANCE

        assert "created_artifacts" in KANBAN_GUIDANCE
        assert "cron.jobs.get_job" in KANBAN_GUIDANCE
        # Issue number named so a future prompt edit that drops the
        # warning trips a unit test before it ships.
        assert "#25288" in KANBAN_GUIDANCE

    def test_guidance_names_idempotency_key(self):
        from agent.prompt_builder import KANBAN_GUIDANCE

        assert "idempotency_key" in KANBAN_GUIDANCE
        # The two natural keys we recommend.
        assert "HERMES_KANBAN_TASK" in KANBAN_GUIDANCE

    def test_do_not_list_warns_about_inventing_artifact_ids(self):
        from agent.prompt_builder import KANBAN_GUIDANCE

        assert "claim an artifact" in KANBAN_GUIDANCE
        assert "without an id you can put in" in KANBAN_GUIDANCE
