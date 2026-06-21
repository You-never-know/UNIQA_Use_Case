"""Fetch all products from FakeStoreAPI and save them to a CSV dataset.

Design notes
------------
Discovery strategy: we never call the bulk GET /products endpoint at all,
including for id discovery. That endpoint returns full product payloads, so
using it just to learn which ids exist would mean fetching every product's
full data twice (once from the bulk call, again from the single-product
call) — wasteful and arguably a stricter reading of the task would consider
it a violation of "use the single-product method" in the first place.

Instead, ids are discovered by probing GET /products/{id} itself, walking
upward from --start-id in batches of --max-concurrency. A 404 response on a
probe IS the discovery signal — there's no separate discovery step, so every
byte fetched is real, kept data. We stop once we've seen
--stop-after-empty-batches consecutive batches where every single id in the
batch came back as a confirmed 404 (never on a single miss, since the API
gives no guarantee ids are contiguous — a lone gap shouldn't be mistaken for
the end of the catalog). A network failure that survives retries is treated
as inconclusive, not as evidence of absence, so it never causes early
termination on its own. --max-id is a hard safety cap in case the API never
returns a clean 404 for some reason.

Other design choices:
- Requests run concurrently with asyncio + aiohttp, bounded by a semaphore
  (also used as the probing batch size) so we don't hammer the API.
- Each response is validated against the Product pydantic model before
  being kept.
- Results are written to CSV in chunks as they arrive (not all at the end),
  so an interrupted run still leaves usable partial data on disk.
- Two storage modes:
    --mode overwrite  -> replace products_processed.csv from scratch
    --mode append      -> add to the existing file, automatically skipping
                          ids that are already saved (safe to re-run after
                          an interruption). Already-saved ids are skipped
                          (no re-fetch) but still count as evidence that
                          region of the id space has real products, so
                          probing correctly continues past them.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import aiohttp
import pandas as pd
from pydantic import ValidationError

from models import COLUMNS, Product

DEFAULT_BASE_URL = "https://fakestoreapi.com"
DEFAULT_OUTPUT = "products_processed.csv"

# Sentinel distinguishing "confirmed does not exist" (404) from a real
# failure (None, after retries) and from success (a flat dict). Confirmed
# absence is the only outcome allowed to count as evidence the id range has
# ended; a real failure is inconclusive and must never be mistaken for that.
NOT_FOUND = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch products from FakeStoreAPI and save them to CSV "
        "using only the single-product endpoint."
    )
    parser.add_argument(
        "--mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="overwrite replaces the CSV; append adds to it and skips ids already saved (default: overwrite)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the output CSV file")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Max concurrent in-flight requests; also the probing batch size",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max attempts per product before giving up"
    )
    parser.add_argument(
        "--retry-backoff", type=float, default=1.0, help="Base seconds for exponential backoff between retries"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5, help="Flush this many fetched products to CSV at a time"
    )
    parser.add_argument(
        "--start-id", type=int, default=1, help="Product id to start probing from"
    )
    parser.add_argument(
        "--stop-after-empty-batches",
        type=int,
        default=1,
        help="Stop once this many consecutive batches are entirely confirmed-404 (no hits, nothing inconclusive)",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        default=10_000,
        help="Hard safety cap on how high to probe, in case the API never returns a clean 404",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("fetch_products")


def load_existing_ids(output_path: Path, logger: logging.Logger) -> set[int]:
    """Return the set of product ids already saved in the CSV, if any."""
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["id"])
        return set(existing["id"].astype(int).tolist())
    except Exception as exc:
        # Could be a permissions issue, a corrupt/partial file, an unexpected
        # schema, etc. Treating it as "nothing saved yet" is a reasonable
        # fallback, but it's surfaced so it isn't mistaken for a clean state.
        logger.warning("Could not read existing ids from %s (%s) — treating as empty.", output_path, exc)
        return set()


async def fetch_product(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    product_id: int,
    max_retries: int,
    backoff_base: float,
    logger: logging.Logger,
) -> dict | None | object:
    """Fetch and validate a single product.

    Returns:
        dict       -> success, flattened product data
        NOT_FOUND  -> confirmed product does not exist
        None       -> network failure, timeout, invalid payload, etc.
    """
    url = f"{base_url}/products/{product_id}"

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:

                    # Explicit 404 = product definitely doesn't exist.
                    if resp.status == 404:
                        return NOT_FOUND

                    resp.raise_for_status()

                    payload = await resp.json()

                # FakeStore sometimes returns:
                # HTTP 200
                # body = null
                #
                # Treat that exactly like a missing product.
                if payload is None:
                    logger.debug(
                        "Product %s returned HTTP 200 with null payload",
                        product_id,
                    )
                    return NOT_FOUND

                # Defensive programming:
                # only dictionaries should be validated as products.
                if not isinstance(payload, dict):
                    logger.error(
                        "Product %s returned unexpected payload type: %s",
                        product_id,
                        type(payload).__name__,
                    )
                    return None

                product = Product.model_validate(payload)
                return product.to_flat_dict()

            except ValidationError as exc:
                logger.error(
                    "Product %s: response failed schema validation: %s",
                    product_id,
                    exc,
                )
                return None

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == max_retries:
                    logger.error(
                        "Product %s: failed after %d attempt(s): %s",
                        product_id,
                        max_retries,
                        exc,
                    )
                    return None

                wait = backoff_base * (2 ** (attempt - 1))

                logger.warning(
                    "Product %s: attempt %d/%d failed (%s), retrying in %.1fs",
                    product_id,
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )

                await asyncio.sleep(wait)

    return None


def initialize_output_file(output_path: Path, mode: str, logger: logging.Logger) -> None:
    """Ensure the CSV is in the correct state before any fetching starts.

    This is what makes overwrite/append behavior deterministic regardless of
    how fetching goes:
    - overwrite mode always truncates here, up front — so the file is
      replaced even if every subsequent fetch fails.
    - append mode only creates a header here if the file doesn't already
      exist. If it does exist, it's left untouched, avoiding a duplicated
      header row mid-file on a second append run.
    Every later write in this run can then safely use header=False.
    """
    if mode == "overwrite":
        pd.DataFrame(columns=COLUMNS).to_csv(output_path, index=False)
        logger.info("Overwrite mode: %s truncated and ready", output_path)
    elif not output_path.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(output_path, index=False)
        logger.info("Append mode: %s did not exist, created with header", output_path)


def flush_chunk(records: list[dict], output_path: Path) -> None:
    """Append a batch of records to the CSV. Assumes the header already exists
    on disk — see initialize_output_file, which guarantees that before any
    flush_chunk call happens."""
    df = pd.DataFrame(records, columns=COLUMNS)
    df.to_csv(output_path, mode="a", header=False, index=False)


async def run(args: argparse.Namespace) -> None:
    logger = setup_logging(args.log_level)
    output_path = Path(args.output)

    existing_ids = load_existing_ids(output_path, logger) if args.mode == "append" else set()
    initialize_output_file(output_path, args.mode, logger)

    logger.info(
        "Probing products from id=%d in batches of %d, via single-product endpoint only "
        "(stopping after %d fully-confirmed-empty batch(es))",
        args.start_id,
        args.max_concurrency,
        args.stop_after_empty_batches,
    )

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(args.max_concurrency)

        buffer: list[dict] = []
        fetched_count = 0
        skipped_count = 0
        not_found_count = 0
        failed_count = 0

        current_id = args.start_id
        consecutive_empty_batches = 0

        while current_id <= args.max_id and consecutive_empty_batches < args.stop_after_empty_batches:
            batch_end = min(current_id + args.max_concurrency, args.max_id + 1)
            batch_ids = list(range(current_id, batch_end))
            current_id = batch_end

            ids_to_probe = [pid for pid in batch_ids if pid not in existing_ids]
            already_known = len(batch_ids) - len(ids_to_probe)
            skipped_count += already_known

            # Starts True; flips to False the moment we see anything other
            # than a confirmed 404 (a known-good id, a success, or an
            # inconclusive failure).
            batch_confirmed_empty = already_known == 0

            if ids_to_probe:
                tasks = [
                    fetch_product(
                        session, semaphore, args.base_url, pid, args.max_retries, args.retry_backoff, logger
                    )
                    for pid in ids_to_probe
                ]
                results = await asyncio.gather(*tasks)

                for pid, outcome in zip(ids_to_probe, results):
                    if outcome is NOT_FOUND:
                        not_found_count += 1
                    elif outcome is None:
                        failed_count += 1
                        batch_confirmed_empty = False  # inconclusive, not evidence of absence
                    else:
                        buffer.append(outcome)
                        fetched_count += 1
                        batch_confirmed_empty = False

            consecutive_empty_batches = consecutive_empty_batches + 1 if batch_confirmed_empty else 0

            if len(buffer) >= args.chunk_size:
                flush_chunk(buffer, output_path)
                buffer = []

        if buffer:
            flush_chunk(buffer, output_path)

        logger.info(
            "Done. Saved %d new product(s) to %s; %d skipped (already saved); "
            "%d confirmed absent; %d failed after retries.",
            fetched_count,
            output_path,
            skipped_count,
            not_found_count,
            failed_count,
        )


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()