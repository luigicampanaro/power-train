import logging

from tensorboard import program


def launch_tensorboard(logdir: str) -> str:
    """TensorBoard on logdir, in a background thread of this process (it stops when the process ends): its URL.
    Port 6006, or the next free one."""
    # Hydra sets the root logger to INFO, so TensorBoard's per-request logs (and --verbose on its Rust data server,
    # which it adds when its logger is at INFO) would flood the training console. Warnings and errors still show.
    logging.getLogger("tensorboard").setLevel(logging.WARNING)
    board = program.TensorBoard()
    board.configure(argv=[None, "--logdir", logdir])
    url = board.launch()
    # After launch: TensorBoard's server setup resets werkzeug's logger to inherit the root level (its request log)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    return url
