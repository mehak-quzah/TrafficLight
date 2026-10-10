import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. TRAFFIC ENVIRONMENT
# =====================================================

def get_level(queue):
    if queue <= 3:
        return 0
    elif queue <= 7:
        return 1
    return 2


def get_state(queues):
    return get_level(queues[0]) * 3 + get_level(queues[1])


def traffic_step(
    queues, action, rng, arrival_a, arrival_b,
    green_seconds, service_rate
):
    queues = queues.copy()
    before = queues.sum()

    arrivals_a = rng.poisson(arrival_a * green_seconds)
    arrivals_b = rng.poisson(arrival_b * green_seconds)

    queues[0] += arrivals_a
    queues[1] += arrivals_b

    departures = min(
        queues[action],
        int(service_rate * green_seconds)
    )
    queues[action] -= departures

    reward = float(before - queues.sum())

    if queues[1 - action] >= 12:
        reward -= 1

    return queues, reward


# =====================================================
# 2. Q-LEARNING ALGORITHM
# =====================================================

def train_agent(
    episodes, steps, alpha, gamma, epsilon_min,
    seed, arrival_a, arrival_b, green_seconds, service_rate
):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((9, 2))
    rewards_history = []

    for episode in range(episodes):
        queues = rng.integers(0, 8, size=2)

        epsilon = max(
            epsilon_min,
            1.0 - episode / max(episodes - 1, 1)
        )

        total_reward = 0

        for _ in range(steps):
            state = get_state(queues)

            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                best_actions = np.flatnonzero(
                    q_table[state] == q_table[state].max()
                )
                action = int(rng.choice(best_actions))

            next_queues, reward = traffic_step(
                queues, action, rng, arrival_a, arrival_b,
                green_seconds, service_rate
            )

            next_state = get_state(next_queues)

            q_table[state, action] += alpha * (
                reward
                + gamma * np.max(q_table[next_state])
                - q_table[state, action]
            )

            queues = next_queues
            total_reward += reward

        rewards_history.append(total_reward)

    return q_table, rewards_history


# =====================================================
# 3. EVALUATE CONTROLLERS
# =====================================================

def evaluate_controller(
    mode, q_table, steps, seed, arrival_a, arrival_b,
    green_seconds, service_rate
):
    rng = np.random.default_rng(seed)
    queues = np.array([2, 2])
    records = []

    for step in range(steps):
        state = get_state(queues)

        if mode == "Q-Learning":
            if np.isclose(q_table[state, 0], q_table[state, 1]):
                action = step % 2
            else:
                action = int(np.argmax(q_table[state]))
        else:
            action = step % 2

        queues, reward = traffic_step(
            queues, action, rng, arrival_a, arrival_b,
            green_seconds, service_rate
        )

        records.append({
            "Step": step + 1,
            "Controller": mode,
            "Road A Queue": int(queues[0]),
            "Road B Queue": int(queues[1]),
            "Total Queue": int(queues.sum()),
            "Green Road": "Road A" if action == 0 else "Road B",
            "Reward": reward
        })

    return pd.DataFrame(records)


# =====================================================
# 4. CONSOLE TESTING
# =====================================================

def run_tests():
    print("\nSMART TRAFFIC LIGHT Q-LEARNING TESTS")
    print("=" * 45)

    passed = 0
    failed = 0

    def test(name, function):
        nonlocal passed, failed

        try:
            function()
            print(f"[PASSED] {name}")
            passed += 1
        except Exception as error:
            print(f"[FAILED] {name}: {error}")
            failed += 1

    def test_levels():
        assert get_level(0) == 0
        assert get_level(3) == 0
        assert get_level(4) == 1
        assert get_level(7) == 1
        assert get_level(8) == 2

    def test_states():
        assert get_state([2, 2]) == 0
        assert get_state([5, 2]) == 3
        assert get_state([9, 10]) == 8

    def test_environment():
        rng = np.random.default_rng(42)
        queues, reward = traffic_step(
            np.array([5, 5]), 0, rng,
            0.35, 0.35, 30, 0.60
        )
        assert len(queues) == 2
        assert np.all(queues >= 0)
        assert np.isfinite(reward)

    def test_training():
        q_table, rewards = train_agent(
            10, 10, 0.1, 0.9, 0.05,
            42, 0.35, 0.35, 30, 0.60
        )
        assert q_table.shape == (9, 2)
        assert len(rewards) == 10
        assert np.all(np.isfinite(q_table))

    def test_comparison():
        q_table, _ = train_agent(
            10, 10, 0.1, 0.9, 0.05,
            42, 0.35, 0.35, 30, 0.60
        )

        learned = evaluate_controller(
            "Q-Learning", q_table, 10, 142,
            0.35, 0.35, 30, 0.60
        )

        fixed = evaluate_controller(
            "Fixed-Time", q_table, 10, 142,
            0.35, 0.35, 30, 0.60
        )

        assert len(learned) == 10
        assert len(fixed) == 10
        assert (learned["Total Queue"] >= 0).all()
        assert (fixed["Total Queue"] >= 0).all()

    test("Traffic queue levels", test_levels)
    test("State generation", test_states)
    test("Traffic simulation", test_environment)
    test("Q-Learning training and Q-table", test_training)
    test("Controller performance comparison", test_comparison)

    print("=" * 45)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")

    if failed == 0:
        print("FINAL RESULT: ALL TESTS PASSED")
    else:
        print("FINAL RESULT: SOME TESTS FAILED")

    print("=" * 45)


# =====================================================
# 5. STREAMLIT APPLICATION
# =====================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Smart Traffic Light RL",
        page_icon="🚦",
        layout="wide"
    )

    st.title("🚦 Smart Traffic Light Optimization")
    st.subheader("Q-Learning Based Traffic Signal Simulation")

    st.write(
        "Train an AI agent to manage traffic queues on two roads "
        "and compare it with a fixed-time traffic signal."
    )

    st.sidebar.header("⚙️ Simulation Settings")

    episodes = st.sidebar.slider(
        "Training Episodes", 100, 1000, 500, step=100
    )
    steps_per_episode = st.sidebar.slider(
        "Steps Per Episode", 25, 200, 100, step=25
    )
    evaluation_steps = st.sidebar.slider(
        "Evaluation Steps", 25, 200, 100, step=25
    )
    arrival_a = st.sidebar.slider(
        "Road A Traffic Arrival Rate", 0.05, 0.80, 0.35, step=0.05
    )
    arrival_b = st.sidebar.slider(
        "Road B Traffic Arrival Rate", 0.05, 0.80, 0.35, step=0.05
    )
    green_seconds = st.sidebar.slider(
        "Green Light Duration (seconds)", 10, 60, 30, step=5
    )
    service_rate = st.sidebar.slider(
        "Vehicles Served Per Second", 0.20, 1.00, 0.60, step=0.05
    )
    seed = st.sidebar.number_input(
        "Random Seed", min_value=0, max_value=99999, value=42
    )

    st.sidebar.caption(
        "Traffic rates are simplified simulation settings, "
        "not real-world measurements."
    )

    if st.button(
        "🚀 Train Agent and Run Simulation",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Training Q-Learning agent..."):
            q_table, rewards = train_agent(
                episodes, steps_per_episode,
                0.1, 0.9, 0.05, int(seed),
                arrival_a, arrival_b,
                green_seconds, service_rate
            )

            learned = evaluate_controller(
                "Q-Learning", q_table, evaluation_steps,
                int(seed) + 100, arrival_a, arrival_b,
                green_seconds, service_rate
            )

            fixed = evaluate_controller(
                "Fixed-Time", q_table, evaluation_steps,
                int(seed) + 100, arrival_a, arrival_b,
                green_seconds, service_rate
            )

            st.session_state["q_table"] = q_table
            st.session_state["rewards"] = rewards
            st.session_state["learned"] = learned
            st.session_state["fixed"] = fixed

        st.success("Training and simulation completed!")

    if "q_table" not in st.session_state:
        st.info(
            "Adjust the settings, then click "
            "'Train Agent and Run Simulation'."
        )
    else:
        q_table = st.session_state["q_table"]
        rewards = st.session_state["rewards"]
        learned = st.session_state["learned"]
        fixed = st.session_state["fixed"]

        results = pd.concat([learned, fixed], ignore_index=True)

        rl_average = learned["Total Queue"].mean()
        fixed_average = fixed["Total Queue"].mean()
        difference = fixed_average - rl_average

        st.header("📊 Performance Comparison")
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Q-Learning Average Queue",
            f"{rl_average:.2f} vehicles"
        )
        col2.metric(
            "Fixed-Time Average Queue",
            f"{fixed_average:.2f} vehicles"
        )
        col3.metric(
            "Queue Difference (Fixed − RL)",
            f"{difference:.2f} vehicles"
        )

        if difference > 0:
            st.write("Q-Learning had a lower average queue in this run.")
        elif difference < 0:
            st.write("Fixed-Time had a lower average queue in this run.")
        else:
            st.write("Both controllers had the same average queue in this run.")

        st.caption(
            "Results come from a simplified simulation and may change "
            "with different settings and random seeds."
        )

        st.header("📈 Traffic Queue Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))

        for name, group in results.groupby("Controller"):
            ax.plot(
                group["Step"],
                group["Total Queue"],
                label=name
            )

        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Total Waiting Vehicles")
        ax.set_title("Total Queue: Q-Learning vs Fixed-Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.header("🧠 Q-Learning Training Progress")
        fig, ax = plt.subplots(figsize=(10, 4))

        window = max(1, len(rewards) // 25)
        average_rewards = pd.Series(rewards).rolling(
            window=window, min_periods=1
        ).mean()

        ax.plot(range(1, len(rewards) + 1), average_rewards)
        ax.set_xlabel("Training Episode")
        ax.set_ylabel("Average Reward")
        ax.set_title("Training Reward Progress")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.header("📋 Learned Q-Table")
        q_df = pd.DataFrame(
            q_table,
            columns=["Serve Road A", "Serve Road B"],
            index=[
                "Low / Low", "Low / Medium", "Low / High",
                "Medium / Low", "Medium / Medium", "Medium / High",
                "High / Low", "High / Medium", "High / High"
            ]
        )
        st.dataframe(q_df.round(2), use_container_width=True)

        st.header("📝 Simulation Results")
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download Results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="traffic_rl_results.csv",
            mime="text/csv"
        )

    st.divider()
    st.caption(
        "Academic MVP only. This simulation does not model pedestrian "
        "crossings, yellow lights, emergency vehicles, or real-world "
        "traffic safety constraints. It must not control actual traffic lights."
    )


# =====================================================
# 6. CHOOSE HOW TO RUN
# =====================================================

if __name__ == "__main__":
    if "streamlit" in sys.modules:
        run_streamlit_app()
    else:
        run_tests()