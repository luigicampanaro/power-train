from tensorboard import program


def launch_tensorboard(logdir: str) -> str:
    """TensorBoard on logdir, in a background thread of this process (it stops when the process ends): its URL.
    Port 6006, or the next free one."""
    board = program.TensorBoard()
    board.configure(argv=[None, "--logdir", logdir])
    return board.launch()
