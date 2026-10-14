# GPU Physics Engine 🚀

A high-performance 2D physics engine built with Rust and wgpu, leveraging GPU compute shaders for massive parallel processing of physics simulations.

[![Feature Demo Video Thumbnail](images/image.png)](https://www.youtube.com/watch?v=TxYFeai6B9Q)

[Demo on YouTube](https://www.youtube.com/watch?v=TxYFeai6B9Q)


## ✨ Features

- **GPU-Accelerated**: All physics computations run on the GPU for maximum performance
- **Spatial Grid Partitioning**: Efficient broad-phase collision detection using GPU-based spatial grids. **Learn more**: [NVIDIA GPU Gems - Broad-Phase Collision Detection](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
- **Verlet Integration**: Stable numerical integration for smooth particle motion
- **Real-time Interaction**: Interactive particle spawning and mouse-based attraction forces
- **Scalable**: Handle millions of particles with high framerates
- **Crust / SPIR-V port**: every compute shader also builds from Rust-subset kernels to SPIR-V without rustc ([details](#-crust--spir-v-kernels-no-rustc-no-wgsl))

## 🎮 Controls

| Input | Action |
|-------|--------|
| `W` or `↑` | Move camera up |
| `A` or `←` | Move camera left |
| `S` or `↓` | Move camera down |
| `D` or `→` | Move camera right |
| `P` | Spawn 100 particles at mouse position |
| `Left Click` | Attract particles to mouse |
| `Mouse Wheel` | Zoom in/out |

## 🚀 Quick Start
### Running the Engine
```bash
cargo run --release
```
### Tests
```
cargo test
```

### Benchmark
The benchmark shows the performance for each of the compute shaders at the end of the execution. It creates `benchmark.json` file that can be visualized at `edge://tracing/` or `chrome://tracing/`. 
```
cargo run --release --features benchmark
```


## 🔧 Implementation Details

### GPU Spatial Grid Partitioning
The engine uses a sophisticated spatial partitioning system that divides the simulation space into a uniform grid. Each particle is assigned to grid cells, dramatically reducing the number of collision checks from O(n²) to approximately O(n*k) in most cases.

**Learn more**: [NVIDIA GPU Gems - Broad-Phase Collision Detection](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)

### GPU Collision Response
All collision detection and response calculations are performed in parallel on the GPU using compute shaders, allowing for real-time simulation of millions of interacting particles.

### Verlet Integration
The engine employs Verlet integration for numerical stability and energy conservation, ensuring smooth and realistic particle motion over time.

## Morton encoding
Every 4 seconds, the particles are sorted using morton codes to improve cache locality. 

### Current Limitations
- **2D Only**: Currently supports 2D simulations
- **Circle Shapes**: Only circular particles are supported at this time

## 🦀 Crust / SPIR-V kernels (no rustc, no WGSL)

Every compute shader also exists as a Rust-subset kernel in [`kernels/`](kernels), compiled straight to
SPIR-V by [Crust](https://github.com/brentharts/crust) without rustc or cargo. The WGSL files stay the
reference; the ports are checked against them.

```bash
python3 build.py --crust ../crust --status        # every WGSL shader and its port status
python3 build.py --crust ../crust                 # compile kernels/*.rs to build/spirv/*.spv
python3 build.py --crust ../crust --check         # ...and run each one against a Python model of the WGSL
python3 build.py --crust ../crust --push-buffer --out DIR   # scalar parameters in a storage buffer (binding 30)
```

`--check` runs the compiled SPIR-V in Crust's reference interpreter (no GPU needed) and compares every
output byte with a plain-Python model of the WGSL, including f32 rounding. `--push-buffer` builds need no
push-constant device feature, which is what wgpu / WebGPU hosts use. Crust (`$CRUST_DIR` or `--crust`) is
the only requirement.

| Kernel file | Entry points | Origin |
|---|---|---|
| `home_cell_ids.rs`, `grid.rs` | `create_home_cell_ids`, `build_cell_ids_array` | ports of the grid shaders |
| `particle_integration.rs`, `rearrange.rs` | `verlet_integration`, `rearrange` | ports |
| `radix_sort.rs`, `prefix_sum.rs` | `build_histogram`, `scatter_keys`, `prefix_sum_of_each_block`, ... | ports (workgroup memory, barriers, atomics) |
| `collision_cell_builder.rs`, `collision_solver.rs` | `count_objects_for_each_chunk`, `build_collision_cells_array`, `solve_collisions` | ports |
| `prowl_world.rs` | `integrate_world`, `tile_collide` | added for [Prowl2D](https://github.com/crustos/Prowl2D): gravity and world box; circles against a grid of solid tiles |
| `prowl_bodies.rs` | `solve_joints`, `solve_collisions_grouped`, `integrate_tagged`, `tile_friction`, `raycast`, `sensor_count` | added for Prowl2D: see below |

### Beyond particles: bodies, joints, queries

`prowl_bodies.rs` adds what a game needs on top of the circle solver, all still built from circles:

- **Joints**: distance constraints between two circles or a circle and a fixed point (rigid, rope, strut,
  spring), solved in colour batches so no circle appears twice in a dispatch. Ropes are chains of them.
- **Groups**: circles of one body (the corner circles of a crate) ignore each other in
  `solve_collisions_grouped`, so a **crate** is a grid of circles held by stiff joints.
- **Tile friction**: slows circles touching a solid tile and absorbs part of the bounce (sand piles, crates stop sliding).
- **Ray casts**: a batch of rays against the tile grid (exact grid walk) and all circles.
- **Sensors**: counts of circles inside boxes or circles, filtered by a per-circle tag.

Prowl2D drives all of this from Python through wgpu-py: see its `GPU.md`.

### Current limits of the kernels

- Circles only; boxes, ropes and joints are circles plus constraints, rounded at the corners.
- Positions are interleaved `f32` pairs, y up; helper functions in the kernel language cannot take buffers.
- The wgpu path of the Python host has been verified only on the reference interpreter so far.

## 📊 Performance

**Test Configuration:**
- **GPU**: AMD Radeon RX 6800 XT (with severe thermal throttling)
- **OS**: Windows 11
- **Build**: Release mode (`--release`)
- **Gravity**: disabled

The engine demonstrates excellent scaling characteristics, maintaining smooth framerates even with millions of active particles thanks to GPU parallelization.
![Performance Graph](./images/performance_graph.png)

| Particles | Frame Time |
|-----------|------------|
| 100,000   | 1.03 ms    |
| 256,000   | 1.28 ms    |
| 512,000   | 1.88 ms    |
| 756,000   | 2.53 ms    |
| 1,000,000 | 3.66 ms    |
| 2,000,000 | 8.417 ms   |
| 3,000,000 | 15.62 ms   |
| 4,000,000 | 22.70 ms   |

## 🔮 Future Enhancements

- Compute shaders optimization
- Rendering optimization
- Improve physics


---

*Built with ❤️ using Rust and wgpu*

## Crust `unity_pack --gpu-physics`

`unity_glue.py` plus `runtime/` (`gpuphys.c/.h`, `gpk.py`) let Crust's `unity_pack` run a packed
Unity project's Rigidbody2D / Collider2D physics on this engine's circle solver instead of Box2D:

```
python3 tools/unity_pack.py MyGame --gpu-physics path/to/gpu-physics-engine -o out
```

`gpk.py` translates the kernels to C with Crust; `gpuphys.c` runs them once per invocation on the CPU
(the sort and prefix sum are plain C) behind a world-units API (`gpuphys.h`); `unity_glue.py` generates
`physics_gpu.c`, which maps the Unity tables onto it. Supported and refused scenes are listed in
Crust's `UNITY_PACK.md` ("GPU physics").
