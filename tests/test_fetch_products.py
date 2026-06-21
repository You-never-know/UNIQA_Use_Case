"""Tests for fetch_products.py: per-request behavior, retries, the batched
discovery/probing loop, and CSV persistence (overwrite/append/resume
semantics, including regression tests for bugs found during review)."""

from __future__ import annotations

import asyncio
import logging

import pandas as pd
import pytest

import fetch_products as fp
from conftest import TransientFailure, make_product_payload
from fetch_products import NOT_FOUND, fetch_product

LOGGER = logging.getLogger("test_fetch_products_processed")


class _Args:
    """Minimal stand-in for the argparse.Namespace that run() expects."""

    def __init__(self, **overrides):
        defaults = dict(
            mode="overwrite",
            output="products_processed.csv",
            base_url="http://fake",
            max_concurrency=10,
            max_retries=2,
            retry_backoff=0.001,
            chunk_size=5,
            start_id=1,
            stop_after_empty_batches=1,
            max_id=10_000,
            log_level="WARNING",
        )
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# fetch_product: success / not-found / validation / retries / concurrency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_product_success(make_fake_session):
    session = make_fake_session({5: make_product_payload(5)})

    result = await fp.fetch_product(
        session, asyncio.Semaphore(1), "http://fake", 5, max_retries=1, backoff_base=0.01, logger=LOGGER
    )

    assert result["id"] == 5
    assert result["rating_count"] == 50
    assert session.calls == [5]


@pytest.mark.asyncio
async def test_fetch_product_not_found_via_null_body(make_fake_session):
    """FakeStoreAPI's real-world behavior: HTTP 200 with a null body for a
    missing id, not a 404."""
    session = make_fake_session({})

    result = await fp.fetch_product(
        session, asyncio.Semaphore(1), "http://fake", 999, max_retries=3, backoff_base=0.01, logger=LOGGER
    )

    assert result is fp.NOT_FOUND
    assert session.calls == [999]  # confirmed-absent must never be retried


@pytest.mark.asyncio
async def test_fetch_product_not_found_via_404(make_fake_session):
    session = make_fake_session({}, not_found_as_404=True)

    result = await fp.fetch_product(
        session, asyncio.Semaphore(1), "http://fake", 999, max_retries=3, backoff_base=0.01, logger=LOGGER
    )

    assert result is fp.NOT_FOUND
    assert session.calls == [999]


@pytest.mark.asyncio
async def test_fetch_product_validation_failure_not_retried(make_fake_session):
    broken_payload = make_product_payload(7)
    del broken_payload["price"]
    session = make_fake_session({7: broken_payload})

    result = await fp.fetch_product(
        session, asyncio.Semaphore(1), "http://fake", 7, max_retries=3, backoff_base=0.01, logger=LOGGER
    )

    assert result is None
    assert session.calls == [7]  # exactly one attempt -- bad data is never retried


@pytest.mark.asyncio
async def test_fetch_product_retries_then_succeeds(make_fake_session):
    session = make_fake_session({11: TransientFailure(times=2, then=make_product_payload(11))})

    result = await fp.fetch_product(
        session, asyncio.Semaphore(1), "http://fake", 11, max_retries=5, backoff_base=0.001, logger=LOGGER
    )

    assert result["id"] == 11
    assert session.calls.count(11) == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_fetch_product_fails_after_exhausting_retries(make_fake_session):
    session = make_fake_session({13: TransientFailure(times=10, then=make_product_payload(13))})

    result = await fp.fetch_product(
        session, asyncio.Semaphore(1), "http://fake", 13, max_retries=3, backoff_base=0.001, logger=LOGGER
    )

    assert result is None
    assert session.calls.count(13) == 3  # stopped at max_retries, never reached recovery

@pytest.mark.asyncio
async def test_semaphore_bounds_concurrency():
    """fetch_product must never let more requests run at once than the
    semaphore allows, regardless of how many tasks are scheduled."""
    active = 0
    max_active = 0

    class SlowResponse:
        def __init__(self, payload):
            self.status = 200
            self._payload = payload

        async def __aenter__(self):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            return self

        async def __aexit__(self, *exc_info):
            nonlocal active
            active -= 1
            return False

        def raise_for_status(self):
            pass

        async def json(self):
            return self._payload

    class SlowSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def get(self, url, timeout=None):
            product_id = int(url.rsplit("/", 1)[-1])
            return SlowResponse(make_product_payload(product_id))

    semaphore = asyncio.Semaphore(3)
    session = SlowSession()
    tasks = [
        fp.fetch_product(session, semaphore, "http://fake", pid, 1, 0.01, LOGGER) for pid in range(1, 11)
    ]
    results = await asyncio.gather(*tasks)

    assert all(r is not None for r in results)
    assert max_active <= 3


# ---------------------------------------------------------------------------
# CSV persistence: initialize_output_file, flush_chunk, load_existing_ids
# ---------------------------------------------------------------------------

def test_initialize_output_file_overwrite_truncates_existing_data(tmp_path):
    output_path = tmp_path / "products_processed.csv"
    output_path.write_text(",".join(fp.COLUMNS) + "\n99,STALE,1.0,d,old,i,1.0,1\n")

    fp.initialize_output_file(output_path, "overwrite", LOGGER)

    df = pd.read_csv(output_path)
    assert df.empty
    assert list(df.columns) == fp.COLUMNS


def test_initialize_output_file_append_leaves_existing_file_untouched(tmp_path):
    output_path = tmp_path / "products_processed.csv"
    original = ",".join(fp.COLUMNS) + "\n1,A,1.0,d,c1,i,4.0,10\n"
    output_path.write_text(original)

    fp.initialize_output_file(output_path, "append", LOGGER)

    assert output_path.read_text() == original


def test_initialize_output_file_append_creates_header_if_missing(tmp_path):
    output_path = tmp_path / "products_processed.csv"
    assert not output_path.exists()

    fp.initialize_output_file(output_path, "append", LOGGER)

    df = pd.read_csv(output_path)
    assert df.empty
    assert list(df.columns) == fp.COLUMNS


def test_flush_chunk_appends_without_touching_header(tmp_path, flat_record):
    output_path = tmp_path / "products_processed.csv"
    fp.initialize_output_file(output_path, "overwrite", LOGGER)

    fp.flush_chunk([flat_record(1)], output_path)
    fp.flush_chunk([flat_record(2)], output_path)

    header_line = ",".join(fp.COLUMNS)
    assert output_path.read_text().count(header_line) == 1
    df = pd.read_csv(output_path)
    assert sorted(df["id"].tolist()) == [1, 2]


def test_load_existing_ids_returns_ids_from_csv(tmp_path, flat_record):
    output_path = tmp_path / "products_processed.csv"
    fp.initialize_output_file(output_path, "overwrite", LOGGER)
    fp.flush_chunk([flat_record(1), flat_record(2)], output_path)

    assert fp.load_existing_ids(output_path, LOGGER) == {1, 2}


def test_load_existing_ids_missing_file_returns_empty_set(tmp_path):
    assert fp.load_existing_ids(tmp_path / "does_not_exist.csv", LOGGER) == set()


def test_load_existing_ids_corrupt_file_returns_empty_set_and_warns(tmp_path, caplog):
    output_path = tmp_path / "products_processed.csv"
    output_path.write_text("not,a,valid,products_processed,csv\n1,2,3,4\n")  # no "id" column

    with caplog.at_level(logging.WARNING):
        ids = fp.load_existing_ids(output_path, LOGGER)

    assert ids == set()
    assert any("Could not read existing ids" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# run(): the full discovery/probing loop, end to end
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_fetches_contiguous_catalog_and_stops(tmp_path, monkeypatch, make_fake_session):
    output_path = tmp_path / "products_processed.csv"
    catalog = {i: make_product_payload(i) for i in range(1, 21)}  # ids 1-20 exist
    session = make_fake_session(catalog)
    monkeypatch.setattr(fp.aiohttp, "ClientSession", lambda: session)

    await fp.run(_Args(output=str(output_path)))

    df = pd.read_csv(output_path)
    assert sorted(df["id"].tolist()) == list(range(1, 21))
    # 20 real hits + exactly one batch (10) of confirmation probes, nothing more
    assert sorted(session.calls) == list(range(1, 31))
    assert len(session.calls) == len(set(session.calls))  # every id probed exactly once

@pytest.mark.asyncio

async def test_run_append_resume_skips_known_ids(tmp_path, monkeypatch, make_fake_session):
    output_path = tmp_path / "products_processed.csv"
    catalog = {i: make_product_payload(i) for i in range(1, 21)}

    session1 = make_fake_session(catalog)
    monkeypatch.setattr(fp.aiohttp, "ClientSession", lambda: session1)
    await fp.run(_Args(output=str(output_path)))

    # Second run, append mode, unchanged catalog: nothing new to add, and
    # none of the 20 already-known ids should be re-fetched.
    session2 = make_fake_session(catalog)
    monkeypatch.setattr(fp.aiohttp, "ClientSession", lambda: session2)
    await fp.run(_Args(mode="append", output=str(output_path)))

    df = pd.read_csv(output_path)
    assert sorted(df["id"].tolist()) == list(range(1, 21))  # no duplicates
    assert not (set(session2.calls) & set(range(1, 21)))  # known ids never re-fetched

@pytest.mark.asyncio

async def test_run_tolerates_gap_within_one_batch(tmp_path, monkeypatch, make_fake_session):
    output_path = tmp_path / "products_processed.csv"
    catalog = {i: make_product_payload(i) for i in range(1, 21) if i != 7}  # id 7 missing

    session = make_fake_session(catalog)
    monkeypatch.setattr(fp.aiohttp, "ClientSession", lambda: session)

    await fp.run(_Args(output=str(output_path), max_concurrency=10))

    df = pd.read_csv(output_path)
    expected_ids = [i for i in range(1, 21) if i != 7]
    assert sorted(df["id"].tolist()) == expected_ids


@pytest.mark.asyncio
async def test_run_overwrite_with_total_failure_still_truncates(tmp_path, monkeypatch, make_fake_session):
    """Regression test: overwrite mode must truncate the file even if every
    single fetch in the run fails to find anything."""
    output_path = tmp_path / "products_processed.csv"
    output_path.write_text(",".join(fp.COLUMNS) + "\n99,STALE,1.0,d,old,i,1.0,1\n")

    session = make_fake_session({})  # empty catalog -- id 1 itself is "not found"
    monkeypatch.setattr(fp.aiohttp, "ClientSession", lambda: session)

    await fp.run(_Args(output=str(output_path), max_concurrency=5))

    df = pd.read_csv(output_path)
    assert df.empty
    assert 99 not in df["id"].tolist()


@pytest.mark.asyncio
async def test_run_append_does_not_duplicate_header_on_second_run(tmp_path, monkeypatch, make_fake_session):
    """Regression test: re-running append mode on an existing file must
    never write a second header row mid-file."""
    output_path = tmp_path / "products_processed.csv"
    catalog = {1: make_product_payload(1), 2: make_product_payload(2)}

    session1 = make_fake_session(catalog)
    monkeypatch.setattr(fp.aiohttp, "ClientSession", lambda: session1)
    await fp.run(_Args(output=str(output_path)))

    catalog[3] = make_product_payload(3)
    session2 = make_fake_session(catalog)
    monkeypatch.setattr(fp.aiohttp, "ClientSession", lambda: session2)
    await fp.run(_Args(mode="append", output=str(output_path)))

    header_line = ",".join(fp.COLUMNS)
    assert output_path.read_text().count(header_line) == 1
    df = pd.read_csv(output_path)
    assert sorted(df["id"].tolist()) == [1, 2, 3]

@pytest.mark.asyncio
async def test_fetch_product_success(make_fake_session, product_payload):
    session = make_fake_session({
        1: product_payload(1)
    })

    result = await fetch_product(
        session=session,
        semaphore=asyncio.Semaphore(1),
        base_url="https://fake.test",
        product_id=1,
        max_retries=3,
        backoff_base=0,
        logger=logging.getLogger("test"),
    )

    assert result["id"] == 1
    assert result["rating_rate"] == 4.0


@pytest.mark.asyncio
async def test_fetch_product_null_payload_returns_not_found(
    make_fake_session,
):
    session = make_fake_session({
        1: None
    })

    result = await fetch_product(
        session=session,
        semaphore=asyncio.Semaphore(1),
        base_url="https://fake.test",
        product_id=1,
        max_retries=3,
        backoff_base=0,
        logger=logging.getLogger("test"),
    )

    assert result is NOT_FOUND


@pytest.mark.asyncio
async def test_fetch_product_404_returns_not_found(
    make_fake_session,
):
    session = make_fake_session(
        {},
        not_found_as_404=True,
    )

    result = await fetch_product(
        session=session,
        semaphore=asyncio.Semaphore(1),
        base_url="https://fake.test",
        product_id=999,
        max_retries=3,
        backoff_base=0,
        logger=logging.getLogger("test"),
    )

    assert result is NOT_FOUND


@pytest.mark.asyncio
async def test_fetch_product_invalid_schema_returns_none(
    make_fake_session,
    product_payload,
):
    payload = product_payload(1)

    del payload["price"]

    session = make_fake_session({
        1: payload
    })

    result = await fetch_product(
        session=session,
        semaphore=asyncio.Semaphore(1),
        base_url="https://fake.test",
        product_id=1,
        max_retries=3,
        backoff_base=0,
        logger=logging.getLogger("test"),
    )

    assert result is None


@pytest.mark.asyncio
async def test_fetch_product_retries_then_succeeds(
    make_fake_session,
    product_payload,
):
    session = make_fake_session({
        1: TransientFailure(
            times=2,
            then=product_payload(1),
        )
    })

    result = await fetch_product(
        session=session,
        semaphore=asyncio.Semaphore(1),
        base_url="https://fake.test",
        product_id=1,
        max_retries=3,
        backoff_base=0,
        logger=logging.getLogger("test"),
    )

    assert result["id"] == 1

    assert session.calls.count(1) == 3


@pytest.mark.asyncio
async def test_fetch_product_fails_after_max_retries(
    make_fake_session,
):
    session = make_fake_session({
        1: TransientFailure(
            times=999
        )
    })

    result = await fetch_product(
        session=session,
        semaphore=asyncio.Semaphore(1),
        base_url="https://fake.test",
        product_id=1,
        max_retries=3,
        backoff_base=0,
        logger=logging.getLogger("test"),
    )

    assert result is None

    assert session.calls.count(1) == 3