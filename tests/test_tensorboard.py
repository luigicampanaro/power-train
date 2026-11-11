import urllib.request

from diff_style.utils.tensorboard import launch_tensorboard


def serves(url: str, logdir: str) -> bool:
    with urllib.request.urlopen(f"{url}data/logdir", timeout=30) as response:
        return response.status == 200 and logdir in response.read().decode()


def test_launch_tensorboard_serves_the_logdir_at_the_url_it_returns(tmp_path):
    assert serves(launch_tensorboard(str(tmp_path)), str(tmp_path))


def test_a_second_tensorboard_takes_the_next_free_port(tmp_path):
    """Two trainings at the same time: the second one does not fail on the port of the first."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = launch_tensorboard(str(tmp_path / "a")), launch_tensorboard(str(tmp_path / "b"))
    assert first != second
    assert serves(first, str(tmp_path / "a")) and serves(second, str(tmp_path / "b"))
