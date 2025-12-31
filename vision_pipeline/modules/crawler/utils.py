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

    # Playwright sync API는 활성화된 asyncio 루프 내에서 실행할 수 없음.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(func, *args, **kwargs).result()
