"""Z-up, fixed-substep interface to a private instance of upstream run.py."""
import math
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/EEF-Cutting-Simulation"
EXAMPLE = UPSTREAM / "configs/example_cutting.yaml"
TO_WORLD = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
ASSET_QUATERNION = np.array([2**-.5, 2**-.5, 0., 0.])


class CuttingEnvironment:
    """The worker owns Taichi's global runtime. Reset/close never reset Kit."""

    def __init__(self):
        self.process = None
        self._channel = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def reset(self, config=EXAMPLE, seed=0, *, show_board=True, knife_control="upstream", fracture=False, tissue_model=None):
        if knife_control not in ("upstream", "external"):
            raise ValueError("knife_control must be 'upstream' or 'external'")
        if fracture and knife_control != "external":
            raise ValueError("Swept fracture requires external knife control")
        config = Path(config or EXAMPLE).resolve(strict=True)
        tissue_path = str(Path(tissue_model).resolve(strict=True)) if tissue_model is not None else ""
        self.close()
        parent, child = socket.socketpair()
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("worker.py")), str(child.fileno()),
                 str(config), str(int(seed)), str(int(show_board)), knife_control, str(int(fracture)), tissue_path],
                cwd=UPSTREAM, pass_fds=(child.fileno(),))
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self._channel = Connection(parent.detach())
        try:
            self.scene = self._receive()
            self.physics_dt = self.scene["dt"]
            self.time = 0.
            self._requested_time = 0.
            self._steps = 0
            return self.get_fruit_state()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        try:
            ok, result = self._channel.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("CulinaryCut worker exited; see worker log") from exc
        if not ok:
            raise RuntimeError(result)
        return result

    def _call(self, command, argument=None):
        if self._channel is None:
            raise RuntimeError("Call reset() first")
        self._channel.send((command, argument))
        return self._receive()

    def step(self, dt):
        """Advance whole upstream ticks; carry the fractional remainder forward."""
        if self._channel is None:
            raise RuntimeError("Call reset() first")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        target = self._requested_time + dt
        new_steps = math.floor(target / self.physics_dt + 1e-9)
        self.time = self._call("advance", new_steps - self._steps)
        self._steps, self._requested_time = new_steps, target
        return self.time

    def get_fruit_state(self):
        """Positions and velocities in world axes, with native damage values."""
        return self._call("fruit")

    def get_tissue_state(self):
        """Read tissue labels and the per-particle material fields used by native P2G."""
        if self._channel is None or "tissues" not in self.scene["fruit"]:
            raise RuntimeError("Reset with a tissue_model before reading material fields")
        return self._call("tissue_state")

    def get_knife_state(self):
        """World pose and twist of the baked knife SDF frame."""
        return self._call("knife")

    def get_contact_force(self):
        """Average material-on-knife grid contact reaction for the most recent step."""
        return self._call("force")

    def get_fracture_state(self):
        if self._channel is None or not self.scene["fracture"]:
            raise RuntimeError("Reset with fracture=True first")
        return self._call("fracture")

    def set_particle_forces(self, forces):
        """Set held world forces in N by particle ID (count × 3), or None to unload."""
        if self._channel is None or self.scene["knife_control"] != "external":
            raise RuntimeError("External knife control is required")
        if forces is not None:
            forces = np.asarray(forces, dtype=np.float32)
            if forces.shape != (self.scene["particle_count"], 3) or not np.isfinite(forces).all():
                raise ValueError("Expected one finite 3D force per particle")
        self._call("particle_forces", forces)

    def set_knife_state(self, position, quaternion, linear_velocity, angular_velocity):
        """Set the knife pose (wxyz) and its world-frame linear and angular velocity."""
        pose = []
        for value, size in zip((position, quaternion, linear_velocity, angular_velocity), (3,4,3,3)):
            array = np.asarray(value, dtype=float)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError("Invalid knife state")
            pose.append(array)
        pos, quat, linear, angular = pose
        if self._channel is None:
            raise RuntimeError("Call reset() first")
        if self.scene["knife_control"] == "external":
            norm = np.linalg.norm(quat)
            if not np.isfinite(norm) or norm < 1e-12:
                raise ValueError("Invalid knife quaternion")
            self._call("set_knife", (pos, quat / norm, linear, angular))
            return
        if not (np.allclose(quat, ASSET_QUATERNION, atol=1e-7, rtol=0) or
                np.allclose(quat, -ASSET_QUATERNION, atol=1e-7, rtol=0)):
            raise ValueError("Upstream knife orientation is fixed")
        if np.any(angular) or np.any(linear[:2]):
            raise ValueError("Upstream boundary velocity supports only vertical translation")
        if linear[2] > 0 and not np.isclose(linear[2], self.scene["return_speed"], rtol=1e-7):
            raise ValueError("Upstream uses its configured speed on the return stroke")
        self._call("set_knife", (pos, float(linear[2])))

    def close(self):
        if self._channel is not None:
            self._channel.close()  # EOF ends the private stepping loop
            self._channel = None
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
