"""Tests for the NFS provider's subfolder mount fallback."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.errors import SetupFailedError

from music_assistant.providers.filesystem_nfs.constants import (
    CONF_EXPORT_PATH,
    CONF_HOST,
    CONF_NFS_VERSION,
    CONF_SUBFOLDER,
)
from music_assistant.providers.filesystem_nfs.provider import NFSFileSystemProvider

INSTANCE_ID = "filesystem_nfs--test"
MOUNT_PATH = f"/tmp/{INSTANCE_ID}"  # noqa: S108
HOST = "nas.local"
EXPORT_PATH = "/mnt/vault"

CHECK_OUTPUT = "music_assistant.providers.filesystem_nfs.provider.check_output"
ISDIR = "music_assistant.providers.filesystem_nfs.provider.isdir"


def _create_provider(subfolder: str | None = None) -> NFSFileSystemProvider:
    """
    Create an NFS provider instance with mocked dependencies.

    :param subfolder: The optional subfolder setup value.
    """
    setup_values: dict[str, Any] = {
        CONF_HOST: HOST,
        CONF_EXPORT_PATH: EXPORT_PATH,
        CONF_NFS_VERSION: "3",
    }
    if subfolder is not None:
        setup_values[CONF_SUBFOLDER] = subfolder

    def _get_setup_value(key: str, default: Any = None) -> Any:
        return setup_values.get(key, default)

    provider = NFSFileSystemProvider.__new__(NFSFileSystemProvider)
    provider.mount_path = MOUNT_PATH
    provider.base_path = MOUNT_PATH
    provider.logger = MagicMock()
    provider.get_setup_value = _get_setup_value  # type: ignore[method-assign]
    return provider


def _mount_sources(mock_check_output: MagicMock) -> list[str]:
    """Return the NFS mount sources of all mount invocations, in call order."""
    return [call.args[-2] for call in mock_check_output.call_args_list if call.args[0] == "mount"]


@pytest.mark.parametrize("subfolder", [None, "", "   ", "/"])
async def test_mount_without_subfolder(subfolder: str | None) -> None:
    """Without a (usable) subfolder the export is mounted as-is in a single attempt."""
    provider = _create_provider(subfolder)
    with patch(CHECK_OUTPUT, AsyncMock(return_value=(0, b""))) as mock_check_output:
        await provider.mount()

    assert _mount_sources(mock_check_output) == [f"{HOST}:{EXPORT_PATH}"]
    assert provider.base_path == MOUNT_PATH


async def test_direct_subfolder_mount_preferred() -> None:
    """When the server allows mounting the subfolder directly, no fallback happens."""
    provider = _create_provider("Music")
    with patch(CHECK_OUTPUT, AsyncMock(return_value=(0, b""))) as mock_check_output:
        await provider.mount()

    assert _mount_sources(mock_check_output) == [f"{HOST}:{EXPORT_PATH}/Music"]
    assert provider.base_path == MOUNT_PATH


async def test_fallback_mounts_export_root_and_scans_subfolder() -> None:
    """A refused subfolder mount falls back to the export root with a local scan root."""
    provider = _create_provider("Music")

    async def fake_check_output(*args: str, **_kwargs: Any) -> tuple[int, bytes]:
        if args[0] != "mount":
            return 0, b""
        # only the bare export is mountable, like an NFSv3 server without -alldirs
        return (0, b"") if args[-2] == f"{HOST}:{EXPORT_PATH}" else (32, b"access denied")

    with (
        patch(CHECK_OUTPUT, AsyncMock(side_effect=fake_check_output)) as mock_check_output,
        patch(ISDIR, AsyncMock(return_value=True)),
    ):
        await provider.mount()

    assert _mount_sources(mock_check_output) == [
        f"{HOST}:{EXPORT_PATH}/Music",
        f"{HOST}:{EXPORT_PATH}",
    ]
    assert provider.mount_path == MOUNT_PATH
    assert provider.base_path == f"{MOUNT_PATH}/Music"


async def test_mount_failure_reports_the_direct_error() -> None:
    """When both mount strategies fail, the direct attempt's error is reported."""
    provider = _create_provider("Music")
    with (
        patch(CHECK_OUTPUT, AsyncMock(return_value=(32, b"access denied by server"))),
        pytest.raises(SetupFailedError, match="access denied by server"),
    ):
        await provider.mount()

    assert provider.base_path == MOUNT_PATH


async def test_fallback_with_missing_subfolder_fails_setup() -> None:
    """A subfolder that does not exist inside the export unmounts again and fails setup."""
    provider = _create_provider("Music")

    async def fake_check_output(*args: str, **_kwargs: Any) -> tuple[int, bytes]:
        if args[0] != "mount":
            return 0, b""
        return (0, b"") if args[-2] == f"{HOST}:{EXPORT_PATH}" else (32, b"access denied")

    with (
        patch(CHECK_OUTPUT, AsyncMock(side_effect=fake_check_output)) as mock_check_output,
        patch(ISDIR, AsyncMock(return_value=False)),
        pytest.raises(SetupFailedError, match="does not exist"),
    ):
        await provider.mount()

    # the failed setup does not leave the export mounted behind
    umount_calls = [
        call.args for call in mock_check_output.call_args_list if call.args[0] == "umount"
    ]
    assert umount_calls == [("umount", MOUNT_PATH)]
    assert provider.base_path == MOUNT_PATH


async def test_traversing_subfolder_is_rejected_before_mounting() -> None:
    """A subfolder that resolves outside the mountpoint is refused without any mount."""
    provider = _create_provider("../../etc")
    with (
        patch(CHECK_OUTPUT, AsyncMock(return_value=(0, b""))) as mock_check_output,
        pytest.raises(SetupFailedError, match="Invalid subfolder"),
    ):
        await provider.mount()

    mock_check_output.assert_not_called()


async def test_unmount_targets_the_mountpoint() -> None:
    """Umount is given the mountpoint even when base_path points inside the mount."""
    provider = _create_provider("Music")
    provider.base_path = f"{MOUNT_PATH}/Music"
    with patch(CHECK_OUTPUT, AsyncMock(return_value=(0, b""))) as mock_check_output:
        await provider.unmount()

    assert mock_check_output.call_args.args == ("umount", MOUNT_PATH)
