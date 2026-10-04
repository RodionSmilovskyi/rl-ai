# Project Context: RL-AI Simulation & Policy Training

You are an expert reinforcement learning and robotics engineer working on the **`rl-ai`** quadcopter simulation and policy training framework.

---

## 1. Mandatory Pre-Execution Requirement: Read `DESIGN.md`

- **MUST READ FIRST:** Before planning, modifying code, refactoring, adding dependencies, or executing any task, you **MUST read [DESIGN.md](file:///home/rodion/projects/rl-ai/DESIGN.md)**.
- **Architectural Alignment:** All code, simulations, and algorithms must strictly adhere to the physical specifications, sensor configurations, and flight control architectures documented in `DESIGN.md`:
  - Quadcopter All-Up Weight: 475g (`DRONE_WEIGHT = 0.475`)
  - Motors: LANNRC 1505 PLUS 3750KV
  - Propellers: 3-inch 3-blade
  - Hover Throttle: 1550 PWM (55% throttle equilibrium)
  - Control loop hierarchy: High-level tactical RL policy (12 Hz) -> Flight controller execution (240 Hz) -> PyBullet simulation environment.
  - Zero-shot sim-to-real transfer parity with `drone-control-center`.

---

## 2. Mandatory Post-Execution Requirement: Keep `DESIGN.md` Updated

- **Synchronize Documentation:** After finishing any code changes, feature additions, bug fixes, or architectural adjustments, you **MUST inspect and update [DESIGN.md](file:///home/rodion/projects/rl-ai/DESIGN.md)**.
- Ensure `DESIGN.md` remains the authoritative single source of truth for:
  - System architecture and control loops
  - Observation and action space specifications
  - Reward formulations and curriculum learning stages
  - Hyperparameters and physical simulation constants
  - Export utilities, ONNX model interfaces, and integration details

---

## 3. Mandatory Post-Execution Requirement: Sync to Google Drive

- **Sync Target:** After finishing execution and updating `DESIGN.md`, both **[DESIGN.md](file:///home/rodion/projects/rl-ai/DESIGN.md)** and **[README.md](file:///home/rodion/projects/rl-ai/README.md)** must be synchronized to the Google Drive **`Drone engineering`** folder.
- **Direct Cloud Sync (No Local Installation Required):**
  Run the automated sync script:
  ```bash
  python scripts/sync_to_gdrive.py
  ```
  - **Cloud Google Drive API:** The script interacts directly with the Google Drive v3 REST API to find or create the `Drone engineering` folder and upload/update both files in cloud storage. It uses `gcloud` access token or `GOOGLE_DRIVE_ACCESS_TOKEN`.
  - **Authorization:** If cloud sync reports insufficient permissions, the user can authorize drive access once via:
    ```bash
    gcloud auth login --enable-gdrive-access
    ```
  - **Local Mount & Staging Fallbacks:** If cloud sync is not authenticated and no local mount is present, the files are staged in `output/google_drive_sync/` and instructions are presented.

---

## 4. Code Modification & Safety Rules

- **Respect Manual Changes:** Always inspect existing code before editing.
- **No Reversions:** Never overwrite or revert user code unless explicitly requested.
- **State Verification:** Verify syntax and test outputs before concluding any task.
