# Project: rl-ai

## Overview
This is a nascent Reinforcement Learning (RL) project. Based on its name and initial environment configuration, it is intended to support the development and training of RL agents.

## Project Structure
- `.python-version`: Specifies the Python environment `rl-ai`.
- `GEMINI.md`: Instructional context for the project.

## Development Setup

### Environment
The project uses a Python environment named `rl-ai`. To ensure consistent execution, verify that this environment is active.

### Expected Files (TODO)
The following files are typically expected in an RL project of this nature but are not yet present:
- `main.py`: Primary entry point for agent training.
- `requirements.txt`: Project dependencies (e.g., `gymnasium`, `stable-baselines3`).
- `README.md`: Project-level documentation.


### Phase 1
## 1. Phase Objective & Executive Summary
The objective of this phase is to build a scalable, multi-CPU reinforcement learning training pipeline. The system will implement the `ParallelSAC` algorithm to solve a foundational continuous control task from the Gymnasium library. The architecture must be designed to run seamlessly both locally and on AWS cloud infrastructure, while strictly adopting the project structure and logging methodology from the existing `object-detection` project.

## 2. Gemini CLI Context & Constraints
*   **Reference Project Integration:** Deeply analyze the `@object-detection` folder. You must strictly adopt its general project structure and exact logging approach for this reinforcement learning phase.

## 3. MoSCoW Prioritization
*   **Must-Have:** `ParallelSAC` algorithm implementation, dynamic multi-CPU scaling, AWS cloud compatibility, integration of the `@object-detection` structure/logging, and video recording of evaluation episodes.


## 4. Functional Requirements (EARS Syntax)
*   **FR1 (Core Algorithm):** The system **shall** implement the `ParallelSAC` algorithm for training the policy.
*   **FR2 (Deployment):** The training architecture **shall** be designed to execute seamlessly both locally and on AWS cloud infrastructure.
*   **FR3 (Performance):** **When** executing training (locally or in the cloud), the system **shall** dynamically leverage the maximum number of available CPUs (e.g., via `os.cpu_count()`) to ensure maximum training speed and performance.
*   **FR4 (Architecture & Logging):** The system **shall** utilize the exact project file structure and logging methodology found in the `@object-detection` reference folder.
*   **FR5 (Evaluation):** **When** an evaluation episode is performed, the system **shall** record the episode rendering as a video and save it directly to a local `output/` directory.

## 5. Step-by-Step Execution Instructions for Gemini CLI
Please execute this phase by following these step-by-step instructions:
1.  **Reasoning:** Think step-by-step about how to structure the multi-processing logic for `ParallelSAC` so that it scales dynamically based on available CPUs while remaining compatible with AWS CPU instances. Show your reasoning.
2.  **Analysis:** Analyze the `@object-detection` folder to extract its architectural blueprint and logging mechanisms.
3.  **Dependencies:** Generate the updated `requirements.txt` file, ensuring any parallel processing or AWS integration libraries are included.
4.  **Training Script:** Generate the core training script, ensuring the environment is set to the simplest Gymnasium continuous control task (e.g., `Pendulum-v1`).
5.  **Clarification Check:** If there are any ambiguities regarding how the `object-detection` logging maps to reinforcement learning metrics (like Actor/Critic loss or Episode Reward), ask me for clarification before writing the code.
