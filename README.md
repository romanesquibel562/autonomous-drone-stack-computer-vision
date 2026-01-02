# Autonomous Drone Perception & Decision System (Work in Progress)

## Overview

This repository contains a work-in-progress autonomous drone control system designed to model a complete real-time autonomy loop:

Perception → Tracking → Decision-Making → Control → Dynamics

The system currently runs in a simulation environment but is architected to transition cleanly to real hardware (e.g., Raspberry Pi, camera modules, sensors) with minimal refactoring. The focus of the project is on building a robust, interpretable autonomy framework before integrating neural-network-based computer vision.

This project emphasizes system design, state management, decision logic, and control flow — core components of real-world autonomous systems.

---

## Project Goals

The long-term objective is to build a real-time autonomous aerial agent capable of:

- Detecting and tracking objects from a live camera feed  
- Prioritizing targets based on confidence, class, and spatial error  
- Making stable control decisions under noisy inputs  
- Operating in real time on embedded hardware  

Current development focuses on autonomy logic and control architecture prior to adding learned perception models.

---

## Current Capabilities

### Simulation
- Deterministic real-time update loop
- Configurable timestep
- Graceful shutdown handling

### Object Tracking
- Persistent track IDs
- Confidence-aware target maintenance
- Track aging and keep/drop logic
- Pixel-space velocity estimation

### Decision Policy
- Mode-based decision system (PATROL, TRACK, AVOID, HOLD)
- Target selection using confidence thresholds
- Deadband-based centering to prevent oscillatory control
- Structured decision outputs with explicit reasoning

### Control & Dynamics
- Command-based control interface (yaw rate, throttle)
- Simple forward-motion dynamics model
- Explicit ego-state representation (position, heading, speed)

### Logging & Observability
- Human-readable per-tick telemetry
- Target ID, label, confidence, and error vectors
- Decision reasoning surfaced for debugging and analysis

---

## Project Architecture
<pre> ```
src/
├── main.py # Entry point (simulation orchestration)
├── core/
│ ├── types.py # Typed dataclasses (Detection, Track, EgoState, Decision)
│ └── init.py
├── perception/
│ ├── sim_perception.py # Simulation perception source
│ └── init.py
├── state_estimation/
│ ├── tracker.py # Tracking and state estimation logic
│ └── init.py
├── decision/
│ ├── policy.py # Decision policy and target selection
│ └── init.py
├── control/
│ ├── init.py # Control interface layer
└── simulation/
├── world.py # Simulated world and dynamics
└── init.py
``` </pre>


### Design Principles
- Strong typing via dataclasses
- Explicit state transitions
- Separation of perception, policy, and control
- Hardware-agnostic interfaces
- Readable, debuggable autonomy logic

---

## Design Rationale

Rather than starting with neural networks, this project builds the autonomy scaffolding first:

- Neural networks are components, not the system
- Control logic must remain interpretable
- Failures should be explainable, not opaque

This mirrors real-world robotics and autonomous systems, where decision logic, safety constraints, and state management are as critical as perception accuracy.

---

## Planned Development Roadmap

### Short-Term
- Priority-weighted target scoring (e.g., person > vehicle > obstacle)
- Adaptive deadband tuning
- Distance estimation integration
- Improved motion dynamics

### Mid-Term
- Live camera input
- Real object detection (e.g., YOLO, MobileNet)
- Multi-object tracking (Kalman filter / SORT-style tracking)
- Real-time frame processing pipeline

### Long-Term
- Raspberry Pi deployment
- Hardware control integration
- Sensor fusion (camera + IMU)
- Learning-based policy experimentation

---

## Technologies Used

- Python 3
- Dataclasses and type hints
- Real-time simulation loops
- State-machine-style policy design
- Control systems fundamentals
- Robotics-oriented software architecture

---

## Project Status

This project is actively under development.

Core autonomy logic and simulation infrastructure are functional and stable. Computer vision and hardware integration are upcoming phases.

The repository reflects real, incremental engineering progress rather than a finished or toy implementation.

---

## Motivation

This project serves both as:
- A learning platform for autonomous systems and robotics
- A portfolio demonstration of system-level engineering beyond model training

It is intentionally designed to scale from simulation to real-world deployment.

## Author:
Roman Esquibel
romanesquib@gmail.com