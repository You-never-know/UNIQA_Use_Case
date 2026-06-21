"""Shared fixtures and test doubles for the products pipeline test suite.

The fetch_products tests never touch the real network. Instead, fetch_products.aiohttp.ClientSession
is monkeypatched to an instance of FakeClientSession below, which simulates the
real FakeStoreAPI closely enough to exercise every code path we care about:
  - a product that exists -> 200 with its JSON payload
  - a product that doesn't exist -> 200 with a JSON `null` body, which is
    FakeStoreAPI's actual real-world behavior (confirmed by hitting the live
    API), NOT a 404 -- though FakeClientSession can also simulate a literal
    404 via not_found_as_404=True, since fetch_product needs to handle both.
  - a product that fails transiently N times before succeeding/resolving,
    via the TransientFailure marker, to exercise the retry/backoff path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root (containing models.py / fetch_products.py /
# analyze_products.py) importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_product_payload(product_id: int, **overrides) -> dict:
    """Build a FakeStoreAPI-shaped product payload (nested `rating` object),
    i.e. what GET /products/{id} actually returns on the wire."""
    payload = {
        "id": product_id,
        "title": f"Product {product_id}",
        "price": float(product_id) + 0.99,
        "description": f"Description for product {product_id}",
        "category": ["electronics", "jewelery", "men's clothing"][product_id % 3],
        "image": f"https://example.com/{product_id}.jpg",
        "rating": {"rate": 4.0, "count": product_id * 10},
    }
    payload.update(overrides)
    return payload


def make_flat_record(product_id: int, **overrides) -> dict:
    """Build a flattened record matching models.COLUMNS, i.e. what
    fetch_product()/Product.to_flat_dict() returns after validation, and
    what flush_chunk()/the CSV layer expects."""
    record = {
        "id": product_id,
        "title": f"Product {product_id}",
        "price": float(product_id) + 0.99,
        "description": f"Description for product {product_id}",
        "category": ["electronics", "jewelery", "men's clothing"][product_id % 3],
        "image": f"https://example.com/{product_id}.jpg",
        "rating_rate": 4.0,
        "rating_count": product_id * 10,
    }
    record.update(overrides)
    return record


class TransientFailure:
    """Catalog marker: simulate `times` consecutive transient request failures
    for this id, then resolve to `then` (a payload dict, or None for
    "not found after recovering")."""

    def __init__(self, times: int, then: dict | None = None):
        self.times = times
        self.then = then


class FakeResponse:
    """Stand-in for an aiohttp response: an async context manager with the
    handful of attributes/methods fetch_product actually uses."""

    def __init__(self, status: int, payload=None):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def raise_for_status(self) -> None:
        import aiohttp

        if self.status >= 400 and self.status != 404:
            raise aiohttp.ClientError(f"simulated HTTP {self.status}")

    async def json(self):
        return self._payload


class FailingResponse:
    """A response whose __aenter__ raises, simulating a transient network
    failure (timeout, connection reset, etc.) at the point fetch_product
    would be inside `async with session.get(...) as resp:`."""

    async def __aenter__(self):
        import aiohttp

        raise aiohttp.ClientError("simulated transient failure")

    async def __aexit__(self, *exc_info):
        return False


_MISSING = object()


class FakeClientSession:
    """Stand-in for aiohttp.ClientSession, backed by an in-memory catalog.

    catalog maps product_id -> one of:
      - a payload dict                    -> 200 with that payload
      - None                              -> 200 with a null body (not found)
      - a TransientFailure(...) instance  -> fails N times, then resolves

    Ids not present in the catalog at all also resolve to "not found"
    (200+null by default, or a literal 404 if not_found_as_404=True), so
    callers only need to list the ids that actually exist.
    """

    def __init__(self, catalog: dict[int, dict | None | TransientFailure] | None = None, not_found_as_404: bool = False):
        self.catalog = dict(catalog or {})
        self.not_found_as_404 = not_found_as_404
        self.calls: list[int] = []
        self._attempt_counts: dict[int, int] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def get(self, url: str, timeout=None):
        product_id = int(url.rsplit("/", 1)[-1])
        self.calls.append(product_id)

        entry = self.catalog.get(product_id, _MISSING)

        if isinstance(entry, TransientFailure):
            attempt = self._attempt_counts.get(product_id, 0) + 1
            self._attempt_counts[product_id] = attempt
            if attempt <= entry.times:
                return FailingResponse()
            entry = entry.then  # fall through to resolve normally

        if entry is _MISSING or entry is None:
            if self.not_found_as_404:
                return FakeResponse(404)
            return FakeResponse(200, None)

        return FakeResponse(200, entry)


@pytest.fixture
def make_fake_session():
    """Factory fixture: make_fake_session({1: payload, 2: None}) -> FakeClientSession"""

    def _make(catalog: dict[int, dict | None | TransientFailure] | None = None, not_found_as_404: bool = False):
        return FakeClientSession(catalog, not_found_as_404=not_found_as_404)

    return _make


@pytest.fixture
def product_payload():
    return make_product_payload


@pytest.fixture
def flat_record():
    return make_flat_record