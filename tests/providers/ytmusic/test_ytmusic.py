"""Test YouTube Music Provider."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError, ServerDisconnectedError
from music_assistant_models.errors import LoginFailed, MediaNotFoundError

from music_assistant.providers.ytmusic import YoutubeMusicProvider
from music_assistant.providers.ytmusic.helpers import shorten_error


@pytest.fixture
def provider() -> YoutubeMusicProvider:
    """Return a YoutubeMusicProvider instance with mocked dependencies."""
    mass = AsyncMock()
    mass.http_session = MagicMock()
    manifest = MagicMock()
    manifest.domain = "ytmusic"
    config = MagicMock()
    config.get_value.return_value = "GLOBAL"
    prov = YoutubeMusicProvider(mass, manifest, config)
    prov._po_token_server_url = "http://localhost:4416"
    return prov


def _ping_context_manager(
    *, response: MagicMock | None = None, exc: Exception | None = None
) -> MagicMock:
    """Build a fake async context manager mimicking aiohttp's session.get()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response, side_effect=exc)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def test_verify_po_token_url_success(provider: YoutubeMusicProvider) -> None:
    """A healthy PO Token server (HTTP 200) verifies successfully."""
    response = MagicMock()
    response.status = 200
    response.raise_for_status = MagicMock()
    provider.mass.http_session.get = MagicMock(  # type: ignore[method-assign]
        return_value=_ping_context_manager(response=response)
    )
    assert await provider._verify_po_token_url() is True


@pytest.mark.parametrize(
    "exc",
    [
        # boot race: the POT container's port accepts TCP before the server is serving,
        # so the ping fails with ServerDisconnectedError (a ClientError, but NOT a
        # ClientConnectorError, which is all the provider used to catch).
        ServerDisconnectedError(),
        # connection refused / DNS failure etc. (the originally-handled family).
        ClientError("connection error"),
        # an explicit/implicit request timeout.
        TimeoutError(),
    ],
)
async def test_verify_po_token_url_transient_failure(
    provider: YoutubeMusicProvider, exc: Exception
) -> None:
    """Transient PO Token server errors return False instead of escaping uncaught."""
    provider.mass.http_session.get = MagicMock(  # type: ignore[method-assign]
        return_value=_ping_context_manager(exc=exc)
    )
    assert await provider._verify_po_token_url() is False


@pytest.fixture
def signed_in_provider(provider: YoutubeMusicProvider) -> YoutubeMusicProvider:
    """Return a provider instance that is ready to talk to the (mocked) YTM API."""
    provider._headers = {"Cookie": "__Secure-3PAPISID=test"}
    provider._yt_user = None
    provider.language = "en"
    provider.available = True
    # call_later is a plain (sync) scheduling call, so it must not be an AsyncMock
    provider.mass.call_later = MagicMock()  # type: ignore[method-assign]
    return provider


# the (huge) KeyError ytmusicapi raises when it is served the signed-out variant
# of a browse response, with the entire API response attached to the message
SIGNED_OUT_KEY_ERROR = KeyError(
    "Unable to find 'twoColumnBrowseResultsRenderer' using path "
    f"['contents', 'twoColumnBrowseResultsRenderer'] on {'x' * 5000}"
)

# the cached variants only add a cache lookup, so test the implementations underneath
get_playlist_impl = YoutubeMusicProvider.get_playlist.__wrapped__  # type: ignore[attr-defined]
get_playlist_tracks_impl = YoutubeMusicProvider.get_playlist_tracks.__wrapped__  # type: ignore[attr-defined]


def _unload_calls(provider: YoutubeMusicProvider) -> MagicMock:
    """Return the mock that captures the (scheduled) unload of the provider."""
    return cast("MagicMock", provider.mass.call_later)


async def test_empty_library_aborts_sync_when_signed_out(
    signed_in_provider: YoutubeMusicProvider,
) -> None:
    """A signed-out session returns an empty library, which must not wipe the MA library."""
    with (
        patch(
            "music_assistant.providers.ytmusic.get_library_playlists",
            AsyncMock(return_value=[]),
        ),
        patch(
            "music_assistant.providers.ytmusic.is_session_authenticated",
            AsyncMock(return_value=False),
        ),
        pytest.raises(LoginFailed),
    ):
        [playlist async for playlist in signed_in_provider.get_library_playlists()]
    # the user must be pointed at the expired cookie
    _unload_calls(signed_in_provider).assert_called_once()


async def test_empty_library_is_allowed_when_signed_in(
    signed_in_provider: YoutubeMusicProvider,
) -> None:
    """A genuinely empty library of a signed-in session is reported as such."""
    with (
        patch(
            "music_assistant.providers.ytmusic.get_library_playlists",
            AsyncMock(return_value=[]),
        ),
        patch(
            "music_assistant.providers.ytmusic.is_session_authenticated",
            AsyncMock(return_value=True),
        ),
    ):
        assert [playlist async for playlist in signed_in_provider.get_library_playlists()] == []
    _unload_calls(signed_in_provider).assert_not_called()


async def test_get_playlist_reports_expired_cookie(
    signed_in_provider: YoutubeMusicProvider,
) -> None:
    """An unparsable playlist response of a signed-out session reports a login failure."""
    with (
        patch(
            "music_assistant.providers.ytmusic.get_playlist",
            AsyncMock(side_effect=SIGNED_OUT_KEY_ERROR),
        ),
        patch(
            "music_assistant.providers.ytmusic.is_session_authenticated",
            AsyncMock(return_value=False),
        ),
        pytest.raises(LoginFailed),
    ):
        # bypass the cache decorator to test the actual implementation
        await get_playlist_impl(signed_in_provider, "LM")


async def test_get_playlist_reports_media_not_found_when_signed_in(
    signed_in_provider: YoutubeMusicProvider,
) -> None:
    """An unparsable playlist response of a signed-in session is a regular lookup failure."""
    with (
        patch(
            "music_assistant.providers.ytmusic.get_playlist",
            AsyncMock(side_effect=SIGNED_OUT_KEY_ERROR),
        ),
        patch(
            "music_assistant.providers.ytmusic.is_session_authenticated",
            AsyncMock(return_value=True),
        ),
        pytest.raises(MediaNotFoundError) as exc_info,
    ):
        await get_playlist_impl(signed_in_provider, "LM")
    # the raw ytmusicapi error embeds the entire response, which must not end up in the error
    assert len(str(exc_info.value)) < 500


async def test_get_playlist_tracks_reports_expired_cookie(
    signed_in_provider: YoutubeMusicProvider,
) -> None:
    """A playlist that can not be loaded due to a signed-out session reports a login failure."""
    with (
        patch(
            "music_assistant.providers.ytmusic.get_playlist",
            AsyncMock(side_effect=SIGNED_OUT_KEY_ERROR),
        ),
        patch(
            "music_assistant.providers.ytmusic.is_session_authenticated",
            AsyncMock(return_value=False),
        ),
        pytest.raises(LoginFailed),
    ):
        await get_playlist_tracks_impl(signed_in_provider, "LM")


async def test_get_playlist_tracks_skips_broken_playlist_when_signed_in(
    signed_in_provider: YoutubeMusicProvider,
) -> None:
    """A single unparsable playlist does not break the sync of the remaining playlists."""
    with (
        patch(
            "music_assistant.providers.ytmusic.get_playlist",
            AsyncMock(side_effect=SIGNED_OUT_KEY_ERROR),
        ),
        patch(
            "music_assistant.providers.ytmusic.is_session_authenticated",
            AsyncMock(return_value=True),
        ),
    ):
        assert await get_playlist_tracks_impl(signed_in_provider, "LM") == []


def test_shorten_error() -> None:
    """A huge ytmusicapi error is truncated, a regular error is left untouched."""
    assert shorten_error(KeyError("short")) == "'short'"
    shortened = shorten_error(SIGNED_OUT_KEY_ERROR)
    assert len(shortened) < 300
    assert shortened.endswith("(truncated)")
