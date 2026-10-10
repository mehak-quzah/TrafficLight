
# 🚦 Smart Traffic Light Optimization Using Q-Learning

**Developed by:** Mehak Quzah

## 📌 Project Overview

Smart Traffic Light Optimization is a Reinforcement Learning project that uses Q-Learning to simulate an adaptive traffic signal system.

Traditional traffic signals follow fixed timings, which may not respond effectively to changing traffic conditions. This project trains an AI agent to select which road should receive the green light based on traffic queue levels.

The goal is to investigate whether Q-Learning can reduce vehicle queues compared with a fixed-time traffic signal in a simulated environment.

## 🎯 Objectives

- Understand the application of Reinforcement Learning in traffic management.
- Implement a Q-Learning algorithm for traffic signal decisions.
- Simulate vehicle queues on two roads.
- Compare Q-Learning with a fixed-time controller.
- Visualize training rewards and traffic queue lengths.

## ✨ Features

- 🚗 Two-road traffic simulation
- 🧠 Q-Learning agent with a Q-table
- 🚦 Adaptive selection of the green-light road
- ⚙️ Adjustable traffic arrival rates and simulation settings
- 📊 Comparison with a fixed-time controller
- 📈 Training reward and queue length graphs
- 📋 Display of learned Q-values
- 📥 Downloadable simulation results in CSV format

## 🛠️ Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Reinforcement Learning
- Q-Learning

## 🧠 How Q-Learning Works

The agent observes traffic conditions, selects a road, receives a reward, and updates its Q-table.

The traffic state is based on three queue levels:

- **Low:** 0–3 vehicles
- **Medium:** 4–7 vehicles
- **High:** 8 or more vehicles

The agent has two possible actions:

- Action 0: Serve Road A
- Action 1: Serve Road B

The reward is based on the change in the total vehicle queue, with a small penalty when the unserved road has a long queue.

Through repeated training episodes, the agent learns action values for different traffic states.

## 📂 Project Structure

```text
TrafficLight/
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/mehak-quzah/TrafficLight.git
```

### 2. Open the project folder

```bash
cd TrafficLight
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 📊 Evaluation

The application compares the two controllers using:

- **Average queue length:** Average number of vehicles waiting during the simulation.
- **Training reward:** Rewards received by the agent during learning.
- **Queue comparison graph:** Shows how the total queue changes over simulation steps.
- **Q-table:** Displays the learned action values for each traffic state.

Results depend on the simulation settings and random seed. Actual performance should be evaluated using the results produced by the application.

## 🚀 Future Improvements

- Simulate more roads and intersections.
- Track individual vehicle waiting times.
- Add more realistic traffic arrival patterns.
- Improve the reward function to balance queue reduction and fairness.
- Explore more advanced Reinforcement Learning algorithms.
- Integrate a realistic traffic simulator for further research.

## ⚠️ Limitations

This project is an educational simulation, not a real-world traffic control system. It does not fully model pedestrian crossings, yellow lights, emergency vehicles, road geometry, or traffic safety constraints.

The results cannot be directly generalized to real traffic without more realistic simulation and testing.

## 👩‍💻 Author

**Mehak Quzah**

GitHub: [@mehak-quzah](https://github.com/mehak-quzah)

Repository: [TrafficLight](https://github.com/mehak-quzah/TrafficLight)

---

*Built as a Reinforcement Learning practice project to explore Q-Learning through traffic signal simulation.*
