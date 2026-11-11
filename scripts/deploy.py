"""Run a trained flow-matching policy on the robot through rosbridge (roslibpy), outside the containers.

Two layouts, chosen with --stack:

owned (the robot's current ros-system, its follower_arms launch file):
    The drivers are owner-aware: they subscribe to their command topics from the start and apply a message only when
    its header.frame_id is the channel's owner. The policy claims the arm and the gripper from the authority registry
    (/authority/claim, owner joint_policy from robot.piper.yaml), ticks at the policy's rate (15 Hz, the training
    episodes'), and sends every action stamped with the owner: /<arm>/arm/joint_command and /<arm>/gripper/command.
    Each tick takes, per input, the message stamped closest to the tick minus 0.03335 s, and only when all of them lie
    within 0.03335 s of it: the policy_server's Syncer, the recorder's synchronizer and the training sets all use it.

synced (the older stack: topic_sync, the pace maker, the command mux):
    Each synced set of topic_sync runs one step, and the action goes to /<arm>/policy/commands.

Before the first step, the script compiles the policy and captures it in one CUDA graph (about a minute). After that
each step replays the graph: about 5 ms for the train.yaml model on an RTX 4070 Ti SUPER, against 30 ms eager
(graphed_policy.py).

From this project folder on the robot PC (the Hugging Face cache must hold the DINOv3 model of the checkpoint):
    uv run --group deploy python scripts/deploy.py --checkpoint outputs/<run>/checkpoints/best-step=0022000.ckpt
"""

import argparse
import base64
import json
import os
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
import roslibpy
import torch

from diff_style.inference.controller import ControlSettings, Controller
from diff_style.inference.graphed_policy import GraphedPolicy, compile_for_inference
from diff_style.inference.policy_loader import load_policy

ARM = os.environ.get("ARM", "right").lower()
CONTROL_PERIOD = 1.0 / 15.0  # seconds: the time budget of one step
# Seconds: the spread a set's messages may have. topic_sync's ApproximateTimeSynchronizer, the recorder's, and the
# policy_server's Syncer all use it, and every set of the training episodes obeys it
SYNC_SLOP = 0.03335
KEEP_S = 1.0  # seconds of each input kept, as the policy_server's Syncer
STALE_S = 1.0  # seconds without a set before the owned stack stops
PACE = "pace"
BUFFERED_SETS = 8  # messages kept per input, to regroup the sets that arrive out of order
NEW_EPISODE_GAP = 1.0  # seconds without a tick: the next tick starts a new episode
LOG_EVERY = 150  # steps, 10 s at 15 Hz
OWNER = "joint_policy"  # robot.piper.yaml: the owner of kind "policy" for each arm
JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
GRIPPER_NAME = "gripper_1"
OLD_JOINT_NAMES = ["piper_1", "piper_2", "piper_3", "piper_4", "piper_5", "piper_6", "gripper_1"]


def main() -> None:
    args = parse_args()
    stack = stack_of(args)
    controller = build_controller(args, stack.cameras)
    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run()
    if args.stack == "owned":
        node = OwnedPolicyNode(client, controller, stack, args.timeout, args.dry_run)
    else:
        node = PolicyNode(client, controller, synced_cameras(args.arm), args.arm)
    print(f"running the policy ({args.stack} stack, channels={stack.channels}, cameras={stack.cameras})")
    node.run()
    client.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="a .ckpt of scripts/train.py (with the EMA weights)")
    parser.add_argument("--stack", choices=["owned", "synced"], default="owned",
                        help="owned: the authority stack of the robot's ros-system (today); synced: topic_sync and "
                             "the pace maker (the older layout)")
    parser.add_argument("--arm", default=ARM, choices=["left", "right"])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="seconds to run, then release the channels and exit; 0 = until Ctrl-C")
    parser.add_argument("--frame-size", type=int, nargs=2, default=[480, 640], metavar=("H", "W"),
                        help="the size of the camera frames: the CUDA graph is captured for it")
    parser.add_argument("--ensemble-offset", type=int, default=0,
                        help="steps the robot is commanded ahead (proleptic ensemble; 0: ACT's). The actions sent are "
                             "the policy's past actions: with an offset, they are ahead of the demonstrations'")
    parser.add_argument("--ensemble-decay", type=float, default=0.13,
                        help="the weight of a chunk i steps old is exp(-decay * i); ml/act's 0.13")
    parser.add_argument("--eager", action="store_true", help="no torch.compile: a slower graph, but no compile")
    parser.add_argument("--dry-run", action="store_true",
                        help="claim nothing and publish nothing: read the inputs, run the policy, print its action")
    return parser.parse_args()


@dataclass(frozen=True)
class Stack:
    """Where the policy of the owned stack reads and writes. The defaults are the robot's profile
    (robot.piper.yaml) and the camera order of episode_0.hdf5: gripper, then main."""

    arm: str
    cameras: tuple[str, ...]
    fps: float = 15.0
    owner: str = OWNER

    @property
    def joint_topic(self) -> str:
        return f"/{self.arm}/piper/joint_states"

    @property
    def gripper_topic(self) -> str:
        return f"/{self.arm}/gripper_joint_state"

    @property
    def input_topics(self) -> tuple[str, ...]:
        return (self.joint_topic, self.gripper_topic, *self.cameras)

    @property
    def channels(self) -> tuple[str, str]:
        return (f"{self.arm}/arm", f"{self.arm}/gripper")

    @property
    def commands(self) -> dict[str, tuple[tuple[str, ...], slice]]:
        """topic -> (names, the part of the action) for each channel's command topic (authority/channels.py)."""
        return {f"/{self.channels[0]}/joint_command": (JOINT_NAMES, slice(0, len(JOINT_NAMES))),
                f"/{self.channels[1]}/command": ((GRIPPER_NAME,), slice(len(JOINT_NAMES), None))}


def stack_of(args: argparse.Namespace) -> Stack:
    """The stack's topics. CAMERAS (comma-separated topics) overrides the cameras, in the training camera order."""
    given = [topic.strip() for topic in os.environ.get("CAMERAS", "").split(",") if topic.strip()]
    side = "r" if args.arm == "right" else "l"
    cameras = tuple(given) if given else (f"/gripper_{side}/image_raw/compressed", "/main/image_raw/compressed")
    return Stack(arm=args.arm, cameras=cameras)


def synced_cameras(arm: str) -> list[str]:
    """The camera names of topic_sync (e.g. right/gripper), in the training order: CAMERAS, or the arm's two."""
    given = [camera.strip() for camera in os.environ.get("CAMERAS", "").split(",") if camera.strip()]
    return given or [f"{arm}/gripper", f"{arm}/main"]


def synced_cameras(arm: str) -> list[str]:
    """The camera names of topic_sync (e.g. right/gripper), in the training order: CAMERAS, or the arm's two."""
    given = [camera.strip() for camera in os.environ.get("CAMERAS", "").split(",") if camera.strip()]
    return given or [f"{arm}/gripper", f"{arm}/main"]


def build_controller(args: argparse.Namespace, cameras: Iterable[str]) -> Controller:
    """The policy of the checkpoint, compiled and captured in its CUDA graph by one step on blank inputs, which the
    controller then forgets."""
    policy, spec = load_policy(args.checkpoint, torch.device("cuda"))
    if len(list(cameras)) != spec.num_cameras:
        raise ValueError(f"the policy takes {spec.num_cameras} cameras, got {list(cameras)}")
    if not args.eager:
        compile_for_inference(policy)
    controller = Controller(GraphedPolicy(policy, spec.image_size),
                            ControlSettings(spec, args.ensemble_offset, args.ensemble_decay))
    print("compiling and capturing the policy...")
    blank_frames = np.zeros((spec.num_cameras, *args.frame_size, 3), dtype=np.uint8)
    controller.step(blank_frames, np.zeros(spec.qpos_dim))
    controller.reset()
    return controller


class OwnedPolicyNode:
    """The authority stack: claim the channels, then one step per tick at the policy's rate, commands stamped with the
    owner. Stops when another owner takes a channel, when the inputs go stale, or at the timeout."""

    def __init__(self, client: roslibpy.Ros, controller: Controller, stack: Stack,
                 timeout_s: float = 0.0, dry_run: bool = False) -> None:
        self.client, self.controller, self.stack = client, controller, stack
        self.timeout_s = timeout_s
        self.dry_run = dry_run
        self.latency = LatencyLog()
        self.inputs = Inputs(stack.input_topics)
        self.stopped: str | None = None
        self.skipped = 0
        self.published = 0
        self.last_set = time.time()  # the last tick with a complete set; before the first, the start
        self.claim_service = roslibpy.Service(client, "/authority/claim", "system_interfaces/srv/Claim")
        self.release_service = roslibpy.Service(client, "/authority/release", "system_interfaces/srv/Release")
        self.command_pubs = {topic: roslibpy.Topic(client, topic, "sensor_msgs/msg/JointState")
                             for topic in stack.commands}
        kinds = {stack.joint_topic: "sensor_msgs/msg/JointState",
                 stack.gripper_topic: "sensor_msgs/msg/JointState"}
        kinds |= dict.fromkeys(stack.cameras, "sensor_msgs/msg/CompressedImage")
        self.input_topics = [roslibpy.Topic(client, topic, kind) for topic, kind in kinds.items()]
        for topic in self.input_topics:
            topic.subscribe(self.receiver(topic.name))
        self.authority_topic = roslibpy.Topic(client, "/authority/state", "std_msgs/msg/String")
        self.authority_topic.subscribe(self.on_authority)

    def receiver(self, topic: str):
        def receive(message: dict) -> None:
            self.inputs.put(topic, message)
        return receive

    def on_authority(self, message: dict) -> None:
        """channel -> {owner, space}, JSON (authority/state_topic.py). Another owner on our channels stops the policy."""
        claims = json.loads(message["data"] or "{}")
        taken = {channel: channel_owner(held) for channel, held in claims.items()
                 if channel in self.stack.channels and channel_owner(held) not in ("", self.stack.owner)}
        if taken:
            self.stopped = f"another owner took {taken}"

    def claim(self) -> str:
        """Claim the channels, without forcing; the answer (within 2 s), or why the claim failed."""
        answer = call_service(self.claim_service, {"channels": list(self.stack.channels),
                                                   "spaces": ["joint", "fraction"],
                                                   "owner": self.stack.owner, "force": False})
        return "" if answer.get("success") else answer.get("message", "the claim was refused")

    def release(self) -> None:
        answer = call_service(self.release_service, {"channels": list(self.stack.channels),
                                                     "owner": self.stack.owner, "force": False})
        print(f"released {list(self.stack.channels)}: {answer.get('message', answer)}")

    def run(self) -> None:
        why = "" if self.dry_run else self.claim()
        if why:
            raise RuntimeError(f"cannot claim {list(self.stack.channels)} as {self.stack.owner}: {why}")
        if self.dry_run:
            print(f"dry run: nothing claimed or published, the policy runs at {self.stack.fps:.0f} Hz")
        else:
            print(f"claimed {list(self.stack.channels)} as {self.stack.owner}; {self.stack.fps:.0f} Hz")
        deadline = time.time() + self.timeout_s if self.timeout_s > 0 else None
        period, next_t = 1.0 / self.stack.fps, time.time()  # wall clock: the same clock as the messages' stamps
        try:
            while self.stopped is None and self.client.is_connected and (deadline is None or time.time() < deadline):
                now = time.time()
                if now < next_t:
                    time.sleep(min(next_t - now, 0.005))
                    continue
                next_t += period
                self.tick(now)
        except KeyboardInterrupt:
            pass
        finally:
            print(f"{'stopped: ' + self.stopped if self.stopped else 'done'}; {self.skipped} ticks without a set")
            if not self.dry_run:
                self.release()
            self.close()

    def tick(self, now: float) -> None:
        snapshot = self.inputs.snapshot(now)
        if snapshot is None:
            self.skipped += 1
            if np.isfinite(self.inputs.age(now)) and now - self.last_set > STALE_S:
                self.stopped = f"no complete set for {now - self.last_set:.1f} s; {self.inputs.ages(now)}"
            return
        self.last_set = now
        start = time.perf_counter()
        frames = np.stack([decode_image(snapshot[topic]) for topic in self.stack.cameras])
        qpos = np.concatenate([values_by_name(snapshot[self.stack.joint_topic], JOINT_NAMES),
                               values_by_name(snapshot[self.stack.gripper_topic], (GRIPPER_NAME,))])
        self.publish(self.controller.step(frames, qpos))
        self.latency.add(time.perf_counter() - start)

    def publish(self, action: np.ndarray) -> None:
        if self.dry_run:
            self.published += 1
            if self.published % int(self.stack.fps) == 0:  # once a second
                print(f"action {np.round(action, 3).tolist()}")
            return
        for topic, (names, part) in self.stack.commands.items():
            self.command_pubs[topic].publish(roslibpy.Message({
                "header": {"stamp": roslibpy.Time.now(), "frame_id": self.stack.owner},
                "name": list(names), "position": [float(value) for value in action[part]],
                "velocity": [], "effort": []}))

    def close(self) -> None:
        for topic in [*self.input_topics, self.authority_topic]:
            topic.unsubscribe()


def call_service(service: roslibpy.Service, fields: dict, timeout_s: float = 2.0) -> dict:
    """A service call, waiting for its answer the way the claim client does (callbacks run on the rosbridge thread)."""
    answer: dict = {}
    done = threading.Event()

    def store(response: dict) -> None:
        answer.update(response)
        done.set()

    service.call(roslibpy.ServiceRequest(fields), store)
    if not done.wait(timeout_s):
        raise RuntimeError(f"{service.name} did not answer within {timeout_s} s")
    return answer


def channel_owner(held) -> str:
    """The owner of a channel in /authority/state: its `{owner, space}` entry, or a bare owner name."""
    return (held.get("owner", "") if isinstance(held, dict) else str(held)) or ""


def values_by_name(message: dict, names: Iterable[str]) -> np.ndarray:
    """The message's positions of these joints, in this order (the drivers name every joint they publish)."""
    positions = dict(zip(message["name"], message["position"]))
    missing = [name for name in names if name not in positions]
    if missing:
        raise ValueError(f"{message['header']['frame_id'] or 'a message'} has no {missing}: it has {list(positions)}")
    return np.array([positions[name] for name in names], dtype=np.float32)


class Inputs:
    """Each input's last second, and the message it gives to a tick: the policy_server's Syncer. The tick's pace is
    0.03335 s behind it (a message stamped up to that after the tick counts), and takes per input the message stamped
    closest to the pace; the set counts only when the pace and all of them lie within 0.03335 s."""

    def __init__(self, topics: Iterable[str]) -> None:
        self.recent: dict[str, deque[dict]] = {topic: deque(maxlen=64) for topic in topics}

    def put(self, topic: str, message: dict) -> None:
        recent = self.recent[topic]
        recent.append(message)
        while recent and stamp(recent[0]) < stamp(message) - KEEP_S:
            recent.popleft()

    def snapshot(self, now: float) -> dict[str, dict] | None:
        if not all(recent for recent in self.recent.values()):
            return None
        pace = now - SYNC_SLOP
        chosen = {topic: min(recent, key=lambda message: abs(stamp(message) - pace))
                  for topic, recent in self.recent.items()}
        times = [pace, *(stamp(message) for message in chosen.values())]
        return chosen if max(times) - min(times) < SYNC_SLOP else None

    def age(self, now: float) -> float:
        """Seconds since the newest message of any input; infinite before the first message."""
        newest = [stamp(recent[-1]) for recent in self.recent.values() if recent]
        return now - max(newest) if newest else float("inf")

    def ages(self, now: float) -> str:
        """The age of each input's newest message, for the message that stops a run."""
        return ", ".join(f"{topic} {now - stamp(recent[-1]) if recent else float('inf'):.1f} s"
                         for topic, recent in self.recent.items())


class PolicyNode:
    """The synced topics of the older stack (topic_sync). Each synced set (SyncedSets) runs one step, and the action
    goes to /<arm>/policy/commands, which the command mux forwards while it is in AUTO mode."""

    def __init__(self, client: roslibpy.Ros, controller: Controller, cameras: Iterable[str], arm: str = ARM) -> None:
        """cameras: the synced camera names of the stack (e.g. left/gripper), in the training order."""
        self.client, self.controller = client, controller
        self.cameras = [camera.strip("/") for camera in cameras]
        self.latency = LatencyLog()
        self.last_tick = -np.inf
        self.action_pub = roslibpy.Topic(client, f"/{arm}/policy/commands", "sensor_msgs/msg/JointState")
        inputs = {"piper": (f"/synced/{arm}/piper/joint_states", "sensor_msgs/msg/JointState"),
                  "gripper": (f"/synced/{arm}/gripper_joint_state", "sensor_msgs/msg/JointState")}
        inputs |= {camera: (f"/synced/{camera}/image_raw/compressed", "sensor_msgs/msg/CompressedImage")
                   for camera in self.cameras}
        self.synced_sets = SyncedSets(list(inputs))
        inputs[PACE] = (f"/synced/{arm}/pace_maker", "sensor_msgs/msg/JointState")
        self.topics = [roslibpy.Topic(client, name, message_type) for name, message_type in inputs.values()]
        for key, topic in zip(inputs, self.topics):
            topic.subscribe(self.receiver(key))

    def receiver(self, key: str):
        def receive(message: dict) -> None:
            synced_set = self.synced_sets.add(key, message)
            if synced_set is not None:
                self.run_step(synced_set)
        return receive

    def run(self) -> None:
        try:
            while self.client.is_connected:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def run_step(self, synced_set: dict[str, dict]) -> None:
        tick = stamp(synced_set[PACE])
        if tick - self.last_tick > NEW_EPISODE_GAP:
            self.controller.reset()
        self.last_tick = tick
        start = time.perf_counter()
        frames = np.stack([decode_image(synced_set[camera]) for camera in self.cameras])  # (C, H, W, 3)
        qpos = np.concatenate([synced_set["piper"]["position"], synced_set["gripper"]["position"]])
        self.publish(self.controller.step(frames, qpos))
        self.latency.add(time.perf_counter() - start)

    def publish(self, action: np.ndarray) -> None:
        self.action_pub.publish(roslibpy.Message({
            "header": {"stamp": roslibpy.Time.now(), "frame_id": "world"},
            "name": OLD_JOINT_NAMES, "position": action.tolist(), "velocity": [], "effort": []}))

    def close(self) -> None:
        for topic in self.topics:
            topic.unsubscribe()


class SyncedSets:
    """Regroups the sets of topic_sync. A training step is one set of its synchronizer: the inputs and the pace tick,
    all within SYNC_SLOP of each other. topic_sync publishes each set as one message per topic, and rosbridge keeps
    the order of each topic, but not across topics (and drops messages: best-effort QoS). So the latest message of an
    input can be of the previous set, 1/15 s old. A set is complete when every input has a message within SYNC_SLOP of
    its pace tick. The oldest complete set is returned; older incomplete sets lost a message and are dropped, as the
    recorder drops them."""

    def __init__(self, inputs: list[str]) -> None:
        self.messages = {key: deque(maxlen=BUFFERED_SETS) for key in inputs}
        self.ticks: deque[dict] = deque(maxlen=BUFFERED_SETS)

    def add(self, key: str, message: dict) -> dict[str, dict] | None:
        """The message of input key (or PACE). Returns the set it completes, {input: message, PACE: tick}, or None."""
        (self.ticks if key == PACE else self.messages[key]).append(message)
        for index, tick in enumerate(self.ticks):
            synced_set = self.set_of(tick)
            if synced_set is not None:
                for _ in range(index + 1):  # this tick, and the incomplete ones before it
                    self.ticks.popleft()
                return synced_set
        return None

    def set_of(self, tick: dict) -> dict[str, dict] | None:
        synced_set = {PACE: tick}
        for key, messages in self.messages.items():
            message = closest(messages, stamp(tick))
            if message is None:
                return None
            synced_set[key] = message
        return synced_set


def closest(messages: Iterable[dict], tick: float) -> dict | None:
    """The message stamped nearest to tick, within SYNC_SLOP; None if there is none."""
    candidates = [message for message in messages if abs(stamp(message) - tick) <= SYNC_SLOP]
    return min(candidates, key=lambda message: abs(stamp(message) - tick), default=None)


class LatencyLog:
    """Prints the mean and the maximum time of the last LOG_EVERY steps (decoding, policy, publishing)."""

    def __init__(self) -> None:
        self.durations: list[float] = []

    def add(self, seconds: float) -> None:
        self.durations.append(seconds)
        if len(self.durations) == LOG_EVERY:
            print(f"step: mean {1e3 * np.mean(self.durations):.1f} ms, max {1e3 * np.max(self.durations):.1f} ms "
                  f"(budget {1e3 * CONTROL_PERIOD:.0f} ms)")
            self.durations.clear()


def stamp(message: dict) -> float:
    return message["header"]["stamp"]["sec"] + message["header"]["stamp"]["nanosec"] * 1e-9


def decode_image(message: dict) -> np.ndarray:
    """A CompressedImage as uint8 (H, W, 3), with the channel order of the training frames: the decoding of the
    converter that produced them (ml/act's convert_mcap_to_hdf5.py, and the container's policy_server._rgb): OpenCV
    gives BGR, converted to RGB only when the format names bgr8.

    The two pipelines differ: v4l2_camera (the tomato_picking recordings) names bgr8, so those frames are RGB (checked:
    red tomatoes in channel 0); gscam (the robot's cameras, and the slot_phone_samsung_return episodes, checked on
    episode_0.hdf5) sends format "jpeg", so those frames are BGR. The script must match the converter of its own
    training data, and this rule matches it for both."""
    image_format = message["format"].lower()
    if "jpeg" in image_format:
        image = imdecode(message, cv2.IMREAD_COLOR)
    elif "png" in image_format:
        image = imdecode(message, cv2.IMREAD_UNCHANGED)[:, :, :3]
    else:
        raise ValueError(f"unsupported image format: {message['format']}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if "bgr8" in image_format else image


def imdecode(message: dict, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(base64.b64decode(message["data"]), np.uint8), flags)
    if image is None:
        raise ValueError(f"cannot decode the {message['format']} image of {message['header']['frame_id']!r}")
    return np.asarray(image)


if __name__ == "__main__":
    main()
