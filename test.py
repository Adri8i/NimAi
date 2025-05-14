from nim import NimAI

def test_get_q_value(ai):
    print("\n--- Testing get_q_value ---")
    state = (0, 0, 0, 2)
    action = (3, 2)
    value = ai.get_q_value(state, action)
    print(f"Q-value for state {state}, action {action}: {value}")


def test_update_q_value(ai):
    print("\n--- Testing update_q_value ---")

    # Setup for test
    state = [0, 0, 0, 2]
    action = (3, 2)
    old_q = ai.get_q_value(tuple(state), action)
    reward = 1  # Example reward
    future_q = 0.8  # Example future Q-value (this would come from best_future_reward)

    # Print old Q-value
    print(f"Old Q-value for {state}, action {action}: {old_q}")

    # Update the Q-value
    ai.update_q_value(state, action, old_q, reward, future_q)

    # Get the new Q-value after update
    new_q = ai.get_q_value(tuple(state), action)
    print(f"New Q-value for {state}, action {action}: {new_q}")


def test_best_future_reward(ai):
    print("\n--- Testing best_future_reward ---")
    state = (0, 0, 0, 2)
    best_future_q = ai.best_future_reward(state)
    print(f"Best future Q-value for state {state}: {best_future_q}")



def test_choose_action(ai):
    print("\n--- Testing choose_action ---")


if __name__ == "__main__":
    ai = NimAI()

    test_get_q_value(ai)
    test_update_q_value(ai)
    test_best_future_reward(ai)
    test_choose_action(ai)

    print("\nAll tests completed.")
