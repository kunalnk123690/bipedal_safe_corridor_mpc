# A Sequential MPC Approach to Reactive Planning for Bipedal Robots Using Safe Corridors

This repository demonstrates reactive bipedal navigation through cluttered
environments in MuJoCo. A sequential model predictive controller (MPC) plans
Digit's footsteps through precomputed convex safe corridors, while a
whole-body quadratic program (QP) tracks the plan and enforces robot dynamics,
contact, friction, and actuator constraints.

Please give us a star and cite our paper if you use this project in your research:

```bibtex
@article{Narkhede2022SequentialMPC,
  author  = {Narkhede, Kunal S. and Kulkarni, Abhijeet M. and
             Thanki, Dhruv A. and Poulakakis, Ioannis},
  title   = {A Sequential {MPC} Approach to Reactive Planning for Bipedal
             Robots Using Safe Corridors in Highly Cluttered Environments},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2022},
  volume  = {7},
  number  = {4},
  pages   = {11831--11838},
  doi     = {10.1109/LRA.2022.3204367}
}
```

## What is included

- Linear inverted pendulum (LIP) footstep planning over a receding horizon
- Discrete control-barrier-function constraints for convex safe corridors
- Alternating-foot reachability and heading constraints
- Optional prediction and avoidance of a moving circular obstacle
- A pointwise whole-body QP controller for Digit
- MuJoCo simulation with obstacle, goal, and footstep visualization
- Precomputed map, goal, and safe-corridor data for the included example

The safe-corridor generation pipeline itself is not included. The supplied
MATLAB files in `Map/` contain corridors generated in advance, so this
repository reproduces the planning and control example rather than every part
of the paper's offline workflow.

## Requirements

The example is intended for Ubuntu Linux and requires:

- Python 3.8 or newer
- A C compiler and `make` for the generated CasADi functions
- An OpenGL-capable display for MuJoCo's interactive viewer

On Ubuntu, the following packages provide the usual build and graphics
prerequisites:

```bash
sudo apt update
sudo apt install build-essential python3-dev python3-venv python3-pip \
  libglfw3 libglfw3-dev libglew-dev libosmesa6-dev
```

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/kunalnk123690/bipedal_safe_corridor_mpc.git
cd bipedal_safe_corridor_mpc
python3 -m venv biped_venv
source biped_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the example

Run the simulation from the repository root because the example loads the
robot model and map using relative paths:

```bash
source biped_venv/bin/activate
python main.py
```

The first run invokes `make` in `DigitModel/CasadiFunctions/Generated/` to
compile the generated C functions into shared objects. Once compilation is
complete, the MuJoCo window opens and Digit follows the sequential corridor
goals until it reaches the final goal or an optimizer fails.

The main experiment parameters are the `MPCData` and `QPData` dictionaries near
the beginning of `main.py`. They control the LIP height and step duration, MPC
horizon and reachability bounds, CBF gain, controller gains, and solver
options.

## How the components fit together

1. `CBFOBS.py` loads obstacles, goals, and safe-corridor half spaces from
   `Map/` and tracks the active corridor.
2. `LIPMPC.py` predicts the touchdown LIP state and solves for relative
   footsteps and headings over the planning horizon.
3. `DIGITQP.py` converts the desired LIP and swing-foot motion into actuator
   torques through a whole-body QP.
4. `main.py` advances MuJoCo, changes stance legs, updates the plan, and sends
   control inputs to the robot.
5. `viewer.py` displays persistent map geometry and per-frame planning markers
   using MuJoCo's passive viewer.

## Repository layout

```text
.
├── main.py                     # Simulation entry point
├── CBFOBS.py                   # Map and safe-corridor management
├── LIPMPC.py                   # Sequential footstep MPC
├── DIGITQP.py                  # Whole-body walking QP
├── angles.py                   # Wrapped-angle arithmetic
├── utils.py                    # State, quaternion, and drawing helpers
├── viewer.py                   # MuJoCo passive-viewer adapter
├── Map/                        # Precomputed obstacles, goals, and corridors
├── DigitModel/                 # Digit XML, meshes, and CasADi functions
└── docs/                       # Sphinx documentation source
```



## License and citation

The project code is licensed under the [BSD 3-Clause License](LICENSE), with
copyright attributed to Kunal S. Narkhede, Abhijeet M. Kulkarni, Dhruv A.
Thanki, and Ioannis Poulakakis. The Digit robot model and associated assets in [`DigitModel/digit-v3.xml`](DigitModel/digit-v3.xml) are provided separately under the
[MIT License](DigitModel/LICENSE) by Agility Robotics.