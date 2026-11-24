"""Private upstream solver process with native or externally driven knife motion."""
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import traceback
import hashlib
import tempfile
from contextlib import ExitStack

import numpy as np

from adapter import ASSET_QUATERNION, ROOT, TO_WORLD, UPSTREAM


def serve(sim, pipe, knife_control, tissues=None):
    def geometry(name, pack):
        if pack is None:
            return None
        return dict(mesh=str((UPSTREAM / sim.cfg[name]["mesh_path"]).resolve()),
                    asset_to_sdf=pack["T_asset_to_world"],
                    offset_y=float(pack.get("render_offset_y", 0.)))

    def knife():
        k = sim.knife
        translation = np.array([-k.x_off[None], k.y[None]-sim._knife_origin[1], k.z_off[None]])
        speed = -k.current_speed[None] if k._down[None] else k.return_speed
        return dict(time=sim.sim_time, position=TO_WORLD@translation, quaternion=ASSET_QUATERNION,
                    linear_velocity=np.array([0., 0., speed]), angular_velocity=np.zeros(3),
                    cut_index=sim.current_cut_idx, cutting_complete=sim.cutting_complete)

    pack = sim.cfg["cutting_mesh_pack"]
    voxel = float(pack["voxel_size"])
    stride = max(1, round(float(sim.cfg["cutting_mesh"].get("particle_spacing", .0033)) / voxel))
    initial = sim.particles.x.to_numpy() @ TO_WORLD.T
    first_y = np.where((pack["sdf"][::stride, ::stride, ::stride] < 0).any(axis=(0, 2)))[0][0] * stride
    lift = float(initial[:, 2].min() - (pack["origin"][1] + first_y * voxel))
    pit = None
    if tissues is not None and "rigid_id" in tissues:
        from rigid_pit import RigidPit
        pit = RigidPit(sim, tissues, lift, TO_WORLD.T @ knife()["position"])
        sim.pit = pit
        transfer_to_particles = sim.g2p
        def transfer_and_project():
            transfer_to_particles()
            pit.exclude_pulp()
        sim.g2p = transfer_and_project
        if knife_control == "upstream":
            transfer_to_grid = sim.p2g
            def constrain_and_transfer():
                sim.limit_native_knife()
                transfer_to_grid()
            sim.p2g = constrain_and_transfer
    control = None
    if knife_control == "external":
        from rigid_knife import RigidKnifeControl
        control = RigidKnifeControl(sim, knife())
    fruit_model = dict(**geometry("cutting_mesh", pack), sdf=pack["sdf"], origin=pack["origin"],
                       voxel=voxel, spacing=voxel*stride, lift=lift, rest=initial)
    if pit is not None:
        fruit_model["soft_sdf"] = sim.soft_sdf
    if tissues is not None:
        from tissues import assign_particle_materials, material_coordinates
        tissues["world_to_asset"] = material_coordinates(pack["T_asset_to_world"], lift, TO_WORLD)
        tissue_ids, segment_ids = assign_particle_materials(sim, tissues, initial)
        fruit_model.update(tissues=tissues, tissue_ids=tissue_ids, segment_ids=segment_ids)
    scene = dict(dt=float(sim.dt), knife_control=knife_control, fracture=hasattr(sim, "fracture"), fruit=fruit_model,
                 particle_count=int(sim.pcount[None]), knife=geometry("knife",sim.knife_pack),
                 board=geometry("board",sim.board_pack), return_speed=sim.knife.return_speed,
                 particle_radius=sim.particle_render_radius, grid=sim.n_grid,
                 solver_source=str(UPSTREAM/"mpmcore/sim.py"))
    if pit is not None:
        scene["rigid_pit"] = dict(vertices=pit.vertices @ TO_WORLD.T, faces=pit.faces,
                                 fixed=True, clearance_m=pit.clearance)
    pipe.send((True, scene))
    start = end = 0.
    reaction_impulse = np.zeros(3)
    while True:
        try:
            cmd, arg = pipe.recv()
        except EOFError:
            break
        if cmd == "advance":
            start = sim.sim_time
            impulse = np.zeros(3)
            for _ in range(arg):
                if control is None:
                    sim.step(sim.sim_time)
                else:
                    control.step()
                impulse += np.asarray(sim.knife_impulse_g[None])
            end = sim.sim_time
            reaction_impulse = -(TO_WORLD @ impulse)
            result = end
        elif cmd == "fruit":
            result = dict(time=sim.sim_time, count=int(sim.pcount[None]),
                          positions=sim.particles.x.to_numpy() @ TO_WORLD.T,
                          velocities=sim.particles.v.to_numpy() @ TO_WORLD.T,
                          damage=sim.particles.D.to_numpy())
            if tissues is not None:
                result.update(tissue_ids=tissue_ids, segment_ids=segment_ids)
        elif cmd == "tissue_state" and tissues is not None:
            result = dict(time=sim.sim_time, tissue_ids=tissue_ids, segment_ids=segment_ids,
                          total_mass_kg=sim._calculate_total_mass(),
                          mass_kg=sim.particles.mass.to_numpy(), volume_m3=sim.particles.vol.to_numpy(),
                          mu_pa=sim.particles.mu0.to_numpy(), lambda_pa=sim.particles.la0.to_numpy())
        elif cmd == "knife":
            result = control.state() if control else knife()
            if pit is not None:
                result.update(pit.state())
                if pit.blocked:
                    result["linear_velocity"] = np.zeros(3)
                    result["angular_velocity"] = np.zeros(3)
        elif cmd == "fracture":
            result = sim.fracture.state()
        elif cmd == "particle_forces":
            control.set_forces(arg)
            result = None
        elif cmd == "set_knife" and control:
            control.command(*arg)
            result = None
        elif cmd == "set_knife":
            pos, speed = arg
            x, y, z = TO_WORLD.T @ pos
            k = sim.knife
            k.x_off[None], k.y[None], k.z_off[None] = -x, y+sim._knife_origin[1], z
            k._down[None] = int(speed <= 0)
            if speed <= 0:
                k.current_speed[None] = -speed
            if pit is not None:
                sim.limit_native_knife(command=True)
            result = None
        elif cmd == "force":
            interval = end-start
            result = dict(force=reaction_impulse/interval if interval else np.zeros(3),
                          impulse=reaction_impulse, interval_start=start, interval_end=end,
                          units="N", frame="world_Z_up", source="upstream knife_impulse_g")
        else:
            raise ValueError(f"Unknown worker request {cmd!r}")
        pipe.send((True, result))
    if sim._out_manager is not None:
        sim._out_manager.finish()


def main():
    fd, config, seed, board, knife_control, fracture, tissue_path = sys.argv[1:]
    with Connection(int(fd)) as pipe, ExitStack() as stack:
        try:
            sys.path.insert(0, str(UPSTREAM))
            import taichi as ti
            import mpmcore.sim as core
            import run as entry
            if int(fracture):
                from fracture import SweptFracture, load_fracture_core
                core = load_fracture_core(stack.enter_context(tempfile.TemporaryDirectory()))
            if knife_control == "external":
                from rigid_knife import RigidKnifeCollider
                core.KnifeCollider = RigidKnifeCollider

            tissue_volume = None
            if tissue_path:
                from tissues import configure_tissue_timestep, load_tissues
                tissue_volume = load_tissues(tissue_path)

            class IsaacSession(core.MPMCuttingSim):
                def __init__(self, cfg, *args, **kwargs):
                    if tissue_volume is not None:
                        configure_tissue_timestep(cfg, tissue_volume)
                    super().__init__(cfg, *args, **kwargs)

                def seed_particles_from_mesh(self, pack, material_cfg, **kwargs):
                    if tissue_volume is not None and "rigid_id" in tissue_volume:
                        from rigid_pit import carve_pit
                        pack = carve_pit(pack, tissue_volume)
                        self.soft_sdf = pack["sdf"]
                    return super().seed_particles_from_mesh(pack, material_cfg, **kwargs)

                def limit_native_knife(self, command=False):
                    k = self.knife
                    requested = np.array([-k.x_off[None], k.y[None]-self._knife_origin[1], k.z_off[None]])
                    p, _ = self.pit.limit(requested, np.eye(3))
                    if self.pit.blocked:
                        k.x_off[None], k.y[None], k.z_off[None] = -p[0], p[1]+self._knife_origin[1], p[2]
                        if not command:
                            k.current_speed[None] = 0.
                            k._down[None] = 1

                def _calculate_total_mass(self):
                    if tissue_volume is None or self.pcount[None] == 0:
                        return super()._calculate_total_mass()
                    return float(self.particles.mass.to_numpy().sum(dtype=np.float64))

                def run(self, *, run_sim=True, max_frames=50_000):
                    super().run(run_sim=True, max_frames=0)  # native seeding + initial knife setup
                    if int(fracture):
                        self.fracture = SweptFracture(self)
                    serve(self, pipe, knife_control, tissue_volume)

            ti.init(arch=ti.cuda, enable_fallback=False, random_seed=int(seed))
            core._TI_INIT = True
            entry.MPMCuttingSim = IsaacSession
            cache = ROOT / ".cache/sdf"
            if Path(config).resolve() != UPSTREAM/"configs/example_cutting.yaml":
                cache /= hashlib.sha256(Path(config).read_bytes()).hexdigest()[:16]
            if tissue_volume and "mesh_variant" in tissue_volume:
                import yaml
                mesh, source_hash = tissue_volume["mesh_variant"]
                cfg = yaml.safe_load(Path(config).read_text())
                requested = Path(cfg["cutting_mesh"]["mesh_path"])
                if hashlib.sha256(requested.read_bytes()).hexdigest() not in (source_hash, str(tissue_volume["mesh_sha256"])):
                    raise ValueError(f"Tissue mesh variant does not belong to {requested}")
                cfg["cutting_mesh"]["mesh_path"] = mesh
                temporary = Path(stack.enter_context(tempfile.TemporaryDirectory()))/"cut.yaml"
                temporary.write_text(yaml.safe_dump(cfg))
                config = str(temporary)
            sys.argv = ["run.py", "--config", config, "--headless", "--run-sim", "--sdf-cache-dir", str(cache)]
            if int(board):
                sys.argv.append("--show-board")
            entry.main()
        except Exception:
            pipe.send((False, traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
