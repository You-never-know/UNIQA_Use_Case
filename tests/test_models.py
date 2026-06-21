"""Tests for the Product/Rating pydantic schema in models.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import COLUMNS, Product


def test_valid_payload_parses_and_flattens(product_payload):
    payload = product_payload(1, price=19.99, rating={"rate": 4.5, "count": 200})

    product = Product.model_validate(payload)
    flat = product.to_flat_dict()

    assert flat["id"] == 1
    assert flat["price"] == 19.99
    assert flat["rating_rate"] == 4.5
    assert flat["rating_count"] == 200
    # The flattened dict's keys must match the canonical column order exactly,
    # since flush_chunk/pandas rely on this for consistent CSV columns.
    assert list(flat.keys()) == COLUMNS


def test_missing_top_level_field_raises_validation_error(product_payload):
    payload = product_payload(2)
    del payload["price"]

    with pytest.raises(ValidationError):
        Product.model_validate(payload)


def test_missing_rating_subfield_raises_validation_error(product_payload):
    payload = product_payload(3, rating={"rate": 4.0})  # missing "count"

    with pytest.raises(ValidationError):
        Product.model_validate(payload)


def test_non_dict_payload_raises_validation_error():
    # This is exactly what fetch_product needs to defend against if a null
    # body ever slipped through to model_validate.
    with pytest.raises(ValidationError):
        Product.model_validate(None)


def test_price_accepts_int_and_coerces_to_float(product_payload):
    payload = product_payload(4, price=20)  # int instead of float in the payload

    product = Product.model_validate(payload)

    assert isinstance(product.price, float)
    assert product.price == 20.0