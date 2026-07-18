"""Test we can parse Open Subsonic models into Music Assistant models."""

import logging
import pathlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import aiofiles
import pytest
from libopensonic.media import (
    AlbumID3,
    AlbumInfo,
    ArtistID3,
    ArtistInfo2,
    Child,
    Lyrics,
    Playlist,
    PodcastChannel,
    PodcastEpisode,
    StructuredLyrics,
)
from music_assistant_models.enums import ConfigEntryType

from music_assistant.providers.opensubsonic import get_config_entries
from music_assistant.providers.opensubsonic.parsers import (
    parse_album,
    parse_artist,
    parse_epsiode,
    parse_playlist,
    parse_podcast,
    parse_structured_lyrics,
    parse_track,
)
from music_assistant.providers.opensubsonic.sonic_provider import OpenSonicProvider

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
ARTIST_FIXTURES = list(FIXTURES_DIR.glob("artists/*.artist.json"))
ALBUM_FIXTURES = list(FIXTURES_DIR.glob("albums/*.album.json"))
PLAYLIST_FIXTURES = list(FIXTURES_DIR.glob("playlists/*.playlist.json"))
PODCAST_FIXTURES = list(FIXTURES_DIR.glob("podcasts/*.podcast.json"))
EPISODE_FIXTURES = list(FIXTURES_DIR.glob("episodes/*.episode.json"))
TRACK_FIXTURES = list(FIXTURES_DIR.glob("tracks/*.track.json"))
LYRICS_FIXTURES = list(FIXTURES_DIR.glob("lyrics/*.lyrics.json"))
STRUCTURED_LYRICS_FIXTURES = list(FIXTURES_DIR.glob("structured-lyrics/*.structured-lyrics.json"))

_LOGGER = logging.getLogger(__name__)


async def test_get_config_entries_exposes_use_get_as_advanced_option() -> None:
    """The Open Subsonic provider exposes the HTTP method as an advanced option."""
    entries = await get_config_entries(None)  # type: ignore[arg-type]
    use_get = next(entry for entry in entries if entry.key == "use_get")

    assert use_get.type == ConfigEntryType.BOOLEAN
    assert use_get.default_value is False
    assert use_get.required is False
    assert use_get.advanced is True


async def test_handle_async_init_passes_use_get_to_connection() -> None:
    """Provider initialization forwards the configured HTTP method to libopensonic."""
    mass = Mock(cache=Mock())
    manifest = Mock(domain="opensubsonic", name="OpenSubsonic")
    config = Mock(name="OpenSubsonic Test", instance_id="opensubsonic-test", enabled=True)
    config.get_value.side_effect = {
        "baseURL": "https://bandcamp.com",
        "username": "user",
        "password": "password",
        "port": 443,
        "path": "api/subsonic",
        "use_get": True,
        "enable_legacy_auth": False,
        "enable_podcasts": False,
        "recommend_favorites": False,
        "recommend_new": False,
        "recommend_played": False,
        "recommendation_count": 10,
        "pagination_size": 200,
        "request_raw_file": True,
        "log_level": "INFO",
    }.get
    connection = AsyncMock()
    connection.ping.return_value = True
    connection.get_open_subsonic_extensions.return_value = []

    with patch(
        "music_assistant.providers.opensubsonic.sonic_provider.SonicConnection",
        return_value=connection,
    ) as connection_class:
        provider = OpenSonicProvider(mass, manifest, config, set())
        await provider.handle_async_init()

    connection_class.assert_called_once_with(
        "https://bandcamp.com",
        username="user",
        password="password",
        legacy_auth=False,
        port=443,
        server_path="api/subsonic",
        app_name="Music Assistant",
        use_get=True,
    )


@pytest.mark.parametrize("example", ARTIST_FIXTURES, ids=lambda val: str(val.stem))
async def test_parse_artists(example: pathlib.Path, snapshot: SnapshotAssertion) -> None:
    """Test we can parse artists."""
    async with aiofiles.open(example) as fp:
        artist = ArtistID3.from_json(await fp.read())

    parsed = parse_artist("xx-instance-id-xx", artist).to_dict()
    # sort external Ids to ensure they are always in the same order for snapshot testing
    parsed["external_ids"].sort()
    assert snapshot == parsed

    # Find the corresponding info file
    example_info = example.with_suffix("").with_suffix(".info.json")
    async with aiofiles.open(example_info) as fp:
        artist_info = ArtistInfo2.from_json(await fp.read())

    parsed = parse_artist("xx-instance-id-xx", artist, artist_info).to_dict()
    # sort external Ids to ensure they are always in the same order for snapshot testing
    parsed["external_ids"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", ALBUM_FIXTURES, ids=lambda val: str(val.stem))
async def test_parse_albums(example: pathlib.Path, snapshot: SnapshotAssertion) -> None:
    """Test we can parse albums."""
    async with aiofiles.open(example) as fp:
        album = AlbumID3.from_json(await fp.read())

    parsed = parse_album(_LOGGER, "xx-instance-id-xx", album).to_dict()
    # sort external Ids and genres to ensure they are always in the same order for snapshot testing
    parsed["external_ids"].sort()
    parsed["metadata"]["genres"].sort()
    assert snapshot == parsed

    # Find the corresponding info file
    example_info = example.with_suffix("").with_suffix(".info.json")
    async with aiofiles.open(example_info) as fp:
        album_info = AlbumInfo.from_json(await fp.read())

    parsed = parse_album(_LOGGER, "xx-instance-id-xx", album, album_info).to_dict()
    # sort external Ids and genres to ensure they are always in the same order for snapshot testing
    parsed["external_ids"].sort()
    parsed["metadata"]["genres"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", PLAYLIST_FIXTURES, ids=lambda val: str(val.stem))
async def test_parse_playlist(example: pathlib.Path, snapshot: SnapshotAssertion) -> None:
    """Test we can parse Playlists."""
    async with aiofiles.open(example) as fp:
        playlist = Playlist.from_json(await fp.read())

    parsed = parse_playlist("xx-instance-id-xx", playlist).to_dict()
    # sort external Ids to ensure they are always in the same order for snapshot testing
    parsed["external_ids"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", PODCAST_FIXTURES, ids=lambda val: str(val.stem))
async def test_parse_podcast(example: pathlib.Path, snapshot: SnapshotAssertion) -> None:
    """Test we can parse Podcasts."""
    async with aiofiles.open(example) as fp:
        podcast = PodcastChannel.from_json(await fp.read())

    parsed = parse_podcast("xx-instance-id-xx", podcast).to_dict()
    # sort external Ids to ensure they are always in the same order for snapshot testing
    parsed["external_ids"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", EPISODE_FIXTURES, ids=lambda val: str(val.stem))
async def test_parse_episode(example: pathlib.Path, snapshot: SnapshotAssertion) -> None:
    """Test we can parse Podcast Episodes."""
    async with aiofiles.open(example) as fp:
        episode = PodcastEpisode.from_json(await fp.read())

    example_channel = example.with_suffix("").with_suffix(".podcast.json")
    async with aiofiles.open(example_channel) as fp:
        channel = PodcastChannel.from_json(await fp.read())

    parsed = parse_epsiode("xx-instance-id-xx", episode, channel).to_dict()
    # sort external Ids to ensure they are always in the same order for snapshot testing
    parsed["external_ids"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", TRACK_FIXTURES, ids=lambda val: str(val.stem))
async def test_parse_track(example: pathlib.Path, snapshot: SnapshotAssertion) -> None:
    """Test we can parse Tracks."""
    async with aiofiles.open(example) as fp:
        song = Child.from_json(await fp.read())

    parsed = parse_track(_LOGGER, "xx-instance-id-xx", song).to_dict()
    # sort external Ids, genres, and performers to ensure they are always in the same
    # order for snapshot testing
    parsed["external_ids"].sort()
    parsed["metadata"]["genres"].sort()
    parsed["metadata"]["performers"].sort()
    assert snapshot == parsed

    example_album = example.with_suffix("").with_suffix(".album.json")
    async with aiofiles.open(example_album) as fp:
        album = AlbumID3.from_json(await fp.read())

    parsed = parse_track(
        _LOGGER, "xx-instance-id-xx", song, parse_album(_LOGGER, "xx-instance-id-xx", album)
    ).to_dict()
    # sort external Ids, genres, and performers to ensure they are always in the same
    # order for snapshot testing
    parsed["external_ids"].sort()
    parsed["metadata"]["genres"].sort()
    parsed["metadata"]["performers"].sort()
    if parsed.get("album"):
        parsed["album"]["external_ids"].sort()
        parsed["album"]["metadata"]["genres"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", LYRICS_FIXTURES, ids=lambda val: str(val.stem))
async def test_lyrics(example: pathlib.Path, snapshot: SnapshotAssertion) -> None:
    """Test that we can handle unstructured lyrics."""
    async with aiofiles.open(example) as fp:
        lyrics = Lyrics.from_json(await fp.read())

    example_track = example.with_suffix("").with_suffix(".track.json")
    async with aiofiles.open(example_track) as fp:
        track = Child.from_json(await fp.read())

    parsed = parse_track(_LOGGER, "xx-instance-id-xx", track, None, (lyrics.value, False)).to_dict()
    parsed["external_ids"].sort()
    parsed["metadata"]["genres"].sort()
    parsed["metadata"]["performers"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", STRUCTURED_LYRICS_FIXTURES, ids=lambda val: str(val.stem))
async def test_structured_lyrics(example: pathlib.Path, snapshot: SnapshotAssertion) -> None:
    """Test that we can handle structured lyrics."""
    async with aiofiles.open(example) as fp:
        lyrics = StructuredLyrics.from_json(await fp.read())

    parsed, _ = parse_structured_lyrics(lyrics)
    assert snapshot == parsed
