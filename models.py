"""Pydantic schema for FakeStoreAPI product responses.

Validating each API response against this schema means a malformed or
unexpected payload is caught and logged immediately, instead of silently
corrupting the CSV dataset downstream.
"""

from pydantic import BaseModel

# Column order used consistently for both the CSV header and every row written.
COLUMNS = [
    "id",
    "title",
    "price",
    "description",
    "category",
    "image",
    "rating_rate",
    "rating_count",
]


class Rating(BaseModel):
    rate: float
    count: int


class Product(BaseModel):
    id: int
    title: str
    price: float
    description: str
    category: str
    image: str
    rating: Rating

    def to_flat_dict(self) -> dict[str, int | float | str]:
        """Flatten the nested `rating` object into top-level columns."""
        return {
            "id": self.id,
            "title": self.title,
            "price": self.price,
            "description": self.description,
            "category": self.category,
            "image": self.image,
            "rating_rate": self.rating.rate,
            "rating_count": self.rating.count,
        }