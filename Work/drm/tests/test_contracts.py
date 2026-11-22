import unittest

from drm.contracts import CONTRACT_VERSION, ErrorCodes, envelope, error_record
from drm.contracts.states import (
    JOB_TERMINAL_STATES,
    OPERATION_TERMINAL_STATES,
    JobState,
    OperationState,
)
from drm.contracts.task_manifest import TaskManifestError, parse, validate

VALID_TASK = {
    "id": "task-1",
    "task_type": "calibration.capture",
    "task_version": 1,
    "max_attempts": 3,
    "timeout_seconds": 600,
}


class EnvelopeTests(unittest.TestCase):
    def test_envelope_carries_contract_version(self):
        doc = envelope({"status": "ok"})
        self.assertEqual(doc["contract"], CONTRACT_VERSION)
        self.assertEqual(doc["status"], "ok")

    def test_error_record_with_details(self):
        doc = error_record(ErrorCodes.DAEMON_UNREACHABLE, "down", url="http://x")
        self.assertEqual(doc["contract"], CONTRACT_VERSION)
        self.assertEqual(doc["error"]["code"], "daemon_unreachable")
        self.assertEqual(doc["error"]["message"], "down")
        self.assertEqual(doc["error"]["details"], {"url": "http://x"})

    def test_error_record_without_details_has_no_details_key(self):
        doc = error_record(ErrorCodes.VALIDATION, "bad input")
        self.assertNotIn("details", doc["error"])


class StateTests(unittest.TestCase):
    def test_job_states_match_spec_section_9(self):
        self.assertEqual(
            {state.value for state in JobState},
            {
                "queued",
                "staging",
                "ready",
                "running",
                "validating",
                "complete",
                "retry_pending",
                "failed",
                "cancelled",
                "awaiting_user",
                "reconciling",
            },
        )

    def test_job_terminal_states(self):
        self.assertEqual(
            JOB_TERMINAL_STATES,
            {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED},
        )
        self.assertNotIn(JobState.RUNNING, JOB_TERMINAL_STATES)
        self.assertNotIn(JobState.AWAITING_USER, JOB_TERMINAL_STATES)

    def test_operation_terminal_states(self):
        self.assertEqual(
            OPERATION_TERMINAL_STATES,
            {OperationState.COMPLETE, OperationState.FAILED, OperationState.CANCELLED},
        )


class TaskManifestTests(unittest.TestCase):
    def test_valid_manifest_passes(self):
        self.assertEqual(validate(VALID_TASK), [])
        self.assertEqual(parse(VALID_TASK), VALID_TASK)

    def test_missing_required_field(self):
        bad = {key: value for key, value in VALID_TASK.items() if key != "timeout_seconds"}
        problems = validate(bad)
        self.assertTrue(any("timeout_seconds" in problem for problem in problems))
        with self.assertRaises(TaskManifestError):
            parse(bad)

    def test_unknown_partition_strategy_rejected(self):
        bad = dict(VALID_TASK, partition="random")
        self.assertTrue(any("partition" in problem for problem in validate(bad)))

    def test_nonpositive_attempts_rejected(self):
        for value in (0, -1, "3", True):
            bad = dict(VALID_TASK, max_attempts=value)
            self.assertTrue(validate(bad), f"max_attempts={value!r} should be rejected")

    def test_non_object_rejected(self):
        self.assertEqual(validate(["not", "an", "object"]), ["task manifest must be a JSON object"])


if __name__ == "__main__":
    unittest.main()
