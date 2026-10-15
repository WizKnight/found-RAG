from uuid import uuid4

from app.cli import reconcile_indexes
from app.services.reconciliation_service import ReconciliationOutcome


def make_outcome(
    *,
    recovered: bool = False,
    superseded: bool = False,
    message: str = "Test outcome",
) -> ReconciliationOutcome:
    return ReconciliationOutcome(
        operation_id=uuid4(),
        document_id=uuid4(),
        recovered=recovered,
        superseded=superseded,
        message=message,
    )


def test_cli_returns_zero_when_all_operations_recover(monkeypatch):
    monkeypatch.setattr(
        reconcile_indexes,
        "reconcile_incomplete_operations",
        lambda load_canonical_chunks: [
            make_outcome(recovered=True),
        ],
    )

    assert reconcile_indexes.main() == 0


def test_cli_returns_zero_when_operations_are_superseded(monkeypatch):
    monkeypatch.setattr(
        reconcile_indexes,
        "reconcile_incomplete_operations",
        lambda load_canonical_chunks: [
            make_outcome(superseded=True),
        ],
    )

    assert reconcile_indexes.main() == 0


def test_cli_returns_nonzero_when_any_operation_fails(monkeypatch):
    monkeypatch.setattr(
        reconcile_indexes,
        "reconcile_incomplete_operations",
        lambda load_canonical_chunks: [
            make_outcome(recovered=True),
            make_outcome(message="Simulated recovery failure"),
        ],
    )

    assert reconcile_indexes.main() == 1


def test_cli_returns_zero_when_no_operations_need_recovery(monkeypatch):
    monkeypatch.setattr(
        reconcile_indexes,
        "reconcile_incomplete_operations",
        lambda load_canonical_chunks: [],
    )

    assert reconcile_indexes.main() == 0


def test_cli_returns_nonzero_when_reconciliation_cannot_start(
    monkeypatch,
):
    def fail_to_start(load_canonical_chunks):
        raise RuntimeError("Journal unavailable")

    monkeypatch.setattr(
        reconcile_indexes,
        "reconcile_incomplete_operations",
        fail_to_start,
    )

    assert reconcile_indexes.main() == 1