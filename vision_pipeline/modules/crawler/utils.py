from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


def run_sync_in_thread_if_event_loop(func: Callable[..., T], *args, **kwargs) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return func(*args, **kwargs)

    # Playwright sync API cannot run inside an active asyncio loop.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(func, *args, **kwargs).result()
