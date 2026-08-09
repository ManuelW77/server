"""
Tests for the device playback failure warnings of the Chromecast provider.

Diagnostic logging for https://github.com/music-assistant/support/issues/5753.

Some Cast devices (notably third-party receivers) crash or throw a media error
while rendering a stream, which the user only sees as playback silently ending.
The player logs a warning for the two observable signatures: losing the socket
connection during playback, and a media session ending with idle reason ERROR.
When the player is configured to allow more than 48 kHz/16 bit audio, the
warning also suggests lowering the sample rates.
"""

from __future__ import annotations

import time
from typing import cast
from unittest.mock import MagicMock

from music_assistant_models.enums import PlaybackState, PlayerType
from pychromecast.socket_client import (
    CONNECTION_STATUS_CONNECTED,
    CONNECTION_STATUS_DISCONNECTED,
    CONNECTION_STATUS_LOST,
)

from music_assistant.providers.chromecast.constants import (
    MASS_APP_ID,
    PLAYBACK_FAILURE_WARN_INTERVAL,
)
from music_assistant.providers.chromecast.player import ChromecastPlayer

SAFE_RATES = [(44100, 16), (48000, 16)]
HIRES_RATES = [(44100, 16), (48000, 16), (96000, 24)]


def _warn(fake: MagicMock, message: str = "Something failed") -> None:
    ChromecastPlayer._warn_device_playback_failure(cast("ChromecastPlayer", fake), message)


def _fake_player(
    *,
    rates: list[tuple[int, int]],
    last_warning: float | None = None,
) -> MagicMock:
    fake = MagicMock()
    fake._last_playback_failure_warning = last_warning
    fake.get_supported_sample_rates.return_value = rates
    return fake


def _connection_status(status: str) -> MagicMock:
    connection_status = MagicMock()
    connection_status.status = status
    return connection_status


def _idle_error_media_status(idle_reason: str | None = "ERROR") -> MagicMock:
    status = MagicMock()
    status.player_state = "IDLE"
    status.player_is_idle = True
    status.player_is_playing = False
    status.player_is_paused = False
    status.idle_reason = idle_reason
    status.content_id = None
    status.current_time = 12.3
    status.adjusted_current_time = 12.3
    return status


def test_warning_without_hires_hint_on_safe_rates() -> None:
    """At the default (safe) sample rates the warning carries no sample rate advice."""
    fake = _fake_player(rates=SAFE_RATES)
    _warn(fake)
    fake.logger.warning.assert_called_once()
    logged = " ".join(str(arg) for arg in fake.logger.warning.call_args.args)
    assert "sample rates" not in logged


def test_warning_with_hires_hint() -> None:
    """With rates beyond 48kHz/16 bit enabled, the warning names them as a suspect."""
    fake = _fake_player(rates=HIRES_RATES)
    _warn(fake)
    fake.logger.warning.assert_called_once()
    args = fake.logger.warning.call_args.args
    assert "sample rates" in args[0]
    assert 96.0 in args
    assert 24 in args


def test_warning_rate_limited() -> None:
    """A second failure within the cooldown does not log again."""
    fake = _fake_player(rates=SAFE_RATES)
    _warn(fake)
    _warn(fake)
    fake.logger.warning.assert_called_once()


def test_warning_repeats_after_cooldown() -> None:
    """A failure after the cooldown has passed logs again."""
    last = time.monotonic() - PLAYBACK_FAILURE_WARN_INTERVAL - 1
    fake = _fake_player(rates=SAFE_RATES, last_warning=last)
    _warn(fake)
    fake.logger.warning.assert_called_once()


def _handle_connection(fake: MagicMock, status: str) -> None:
    ChromecastPlayer._handle_connection_status(
        cast("ChromecastPlayer", fake), _connection_status(status)
    )


def _fake_connected_player(
    *,
    playback_state: PlaybackState,
    active_cast_group: str | None = None,
) -> MagicMock:
    fake = MagicMock()
    fake.active_cast_group = active_cast_group
    fake._attr_playback_state = playback_state
    fake.on_app_status_changed = None
    return fake


def test_connection_lost_during_playback_warns() -> None:
    """Losing the connection while playing raises the playback failure warning."""
    for status in (CONNECTION_STATUS_LOST, CONNECTION_STATUS_DISCONNECTED):
        fake = _fake_connected_player(playback_state=PlaybackState.PLAYING)
        _handle_connection(fake, status)
        fake._warn_device_playback_failure.assert_called_once()


def test_connection_lost_while_idle_does_not_warn() -> None:
    """A disconnect without active playback is a regular event, not a failure."""
    fake = _fake_connected_player(playback_state=PlaybackState.IDLE)
    _handle_connection(fake, CONNECTION_STATUS_LOST)
    fake._warn_device_playback_failure.assert_not_called()


def test_connection_lost_on_group_child_does_not_warn() -> None:
    """A group member mirrors its group; only the group itself reports the failure."""
    fake = _fake_connected_player(
        playback_state=PlaybackState.PLAYING,
        active_cast_group="7ef39234-2b22-4569-b460-3e339b297d68",
    )
    _handle_connection(fake, CONNECTION_STATUS_LOST)
    fake._warn_device_playback_failure.assert_not_called()


def test_connection_established_does_not_warn() -> None:
    """A (re)connect never raises the playback failure warning."""
    fake = _fake_connected_player(playback_state=PlaybackState.PLAYING)
    _handle_connection(fake, CONNECTION_STATUS_CONNECTED)
    fake._warn_device_playback_failure.assert_not_called()


def _fake_media_player(*, playback_state: PlaybackState) -> MagicMock:
    fake = MagicMock()
    fake.active_cast_group = None
    fake._attr_playback_state = playback_state
    fake.cc.app_id = MASS_APP_ID
    fake.type = PlayerType.PLAYER
    return fake


def test_media_error_during_playback_warns() -> None:
    """A media session ending with idle reason ERROR raises the failure warning."""
    fake = _fake_media_player(playback_state=PlaybackState.PLAYING)
    ChromecastPlayer._handle_media_status(
        cast("ChromecastPlayer", fake), _idle_error_media_status()
    )
    fake._warn_device_playback_failure.assert_called_once()


def test_media_finished_does_not_warn() -> None:
    """A media session ending normally does not raise the failure warning."""
    fake = _fake_media_player(playback_state=PlaybackState.PLAYING)
    ChromecastPlayer._handle_media_status(
        cast("ChromecastPlayer", fake), _idle_error_media_status(idle_reason="FINISHED")
    )
    fake._warn_device_playback_failure.assert_not_called()
