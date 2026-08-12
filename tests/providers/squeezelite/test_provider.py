"""Tests for the Squeezelite player provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from music_assistant.providers.squeezelite.constants import CONF_DISCOVERY
from music_assistant.providers.squeezelite.provider import SqueezelitePlayerProvider


def _provider_with_mocks(*, discovery_enabled: bool) -> tuple[SqueezelitePlayerProvider, MagicMock]:
    """Return an uninitialized provider alongside its mocked slimproto server."""
    provider = SqueezelitePlayerProvider.__new__(SqueezelitePlayerProvider)
    provider.logger = MagicMock()
    provider.mass = MagicMock()
    provider.config = MagicMock()
    provider.config.get_value.side_effect = lambda key, default=None: (
        discovery_enabled if key == CONF_DISCOVERY else default
    )
    slimproto = MagicMock()
    slimproto.start = AsyncMock()
    provider.slimproto = slimproto
    return provider, slimproto


async def test_discovery_broadcaster_stopped_when_disabled() -> None:
    """The discovery broadcaster is shut down again when discovery is disabled."""
    provider, slimproto = _provider_with_mocks(discovery_enabled=False)
    discovery = slimproto._discovery

    await provider.loaded_in_mass()

    discovery.close.assert_called_once()
    assert slimproto._discovery is None


async def test_discovery_broadcaster_kept_when_enabled() -> None:
    """The discovery broadcaster keeps running when discovery is enabled (the default)."""
    provider, slimproto = _provider_with_mocks(discovery_enabled=True)
    discovery = slimproto._discovery

    await provider.loaded_in_mass()

    discovery.close.assert_not_called()
    assert slimproto._discovery is discovery


def test_stop_discovery_broadcaster_is_idempotent() -> None:
    """Stopping an already stopped discovery broadcaster is a no-op."""
    provider, slimproto = _provider_with_mocks(discovery_enabled=False)
    slimproto._discovery = None

    provider._stop_discovery_broadcaster()

    assert slimproto._discovery is None
