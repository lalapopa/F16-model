import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from tqdm import tqdm
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
RUN_TIME = datetime.now().strftime("%y-%m-%d-%H-%M")
import tensorflow as tf

from agent import model
from agent import dhp as DHP
from make_enviroment import Env
from F16model.model import States, Control, interface
from F16model.model.engine import find_correct_thrust_position


class RefSignalGenerator:
    def __init__(self, n, init_state):
        self.step_numbers = n
        self.init_state = init_state

    def get_reference(self):
        return self.init_state


def generate_random_excitation(n_samples):
    array = []
    for i in range(0, n_samples):
        first_value, second_value = np.random.rand(2)
        random_sign = np.power(-1, np.random.randint(1, 3))
        array.append([random_sign * first_value * 0.2, second_value])
    return array


def run_train(env, agent, ac_model, STATE_ERROR_WEIGHTS, TRACKED):
    X, U, R, C_real = [], [], [], []
    X_pred, C_trained, action_grad, critic_grad = [], [], [], []
    list_F, list_G, list_RLS_cov, e_model = [], [], [], []
    U_ref = []
    total_steps = 1
    max_episode = 5
    max_steps = 4000
    excitation_steps = 105
    nan_occurs = False
    excitation_signal = generate_random_excitation(excitation_steps)
    with tqdm(range(max_episode)) as tqdm_it:
        for i in tqdm_it:
            # init params
            if nan_occurs:
                break
            init_condition = env.reset()
            ac_model.reset()
            ref_generator = RefSignalGenerator(max_steps, init_condition)
            x = init_condition
            P = np.diag(TRACKED).astype(float)
            Q = np.diag(STATE_ERROR_WEIGHTS)
            X.append(init_condition)
            print(f"First init condition {init_condition}")
            done = False
            episode_steps = 1
            while not done:
                x_ref = ref_generator.get_reference().reshape([1, -1, 1])
                R_sig = np.squeeze(x_ref)[1:].reshape([1, -1, 1])
                U_ref.append(R_sig)
                j = 0
                x = x.reshape([1, -1, 1])
                while j < 2:
                    # Next state prediction
                    action = agent.action(x, reference=R_sig).reshape([1, -1, 1])
                    action_clipped = np.clip(
                        action,
                        np.array([[np.radians(-25)], [0.2]]),
                        np.array([[np.radians(25)], [1.0]]),
                    )

                    x_next_pred = ac_model.predict(x, action_clipped).reshape(
                        [1, -1, 1]
                    )

                    if np.isnan(x_next_pred).any():
                        done = True
                        nan_occurs = True
                        print("=" * 8, i, j, f"in {episode_steps}", "=" * 8)
                        print(f"{x_next_pred = }\n{action =}\n{R_sig = }\n{x =}")
                        print("=" * 20)
                        break
                    # Cost prediction
                    e = np.matmul(P, x_next_pred - x_ref)
                    cost = np.matmul(np.matmul(e.transpose(0, 2, 1), Q), e)
                    dcostdx = np.matmul(2 * np.matmul(e.transpose(0, 2, 1), Q), P)

                    dactiondx = agent.gradient_actor(x, reference=R_sig)
                    lmbda = agent.value_derivative(x, reference=R_sig)

                    # Critic
                    target_lmbda = agent.target_value_derivative(
                        x_next_pred, reference=R_sig
                    )
                    A = ac_model.gradient_state(x, action)
                    B = ac_model.gradient_action(x, action)
                    # print(
                    #     f"|ITER {i}||{j}| Values before grad_critic:\n{lmbda =}\n{dcostdx = }\n{agent.gamma =}\n{target_lmbda =}\n{A =}\n{B = }\n{dactiondx =}"
                    # )
                    grad_critic = lmbda - np.matmul(
                        dcostdx + agent.gamma * target_lmbda,
                        A + np.matmul(B, dactiondx),
                    )
                    grad_critic = np.clip(grad_critic, -0.2, 0.2)
                    agent.update_critic(x, reference=R_sig, gradient=grad_critic)
                    # print(f"TOTAL GRAD CRITIC = {grad_critic}")

                    # Actor
                    lmbda = agent.value_derivative(x_next_pred, reference=R_sig)
                    grad_actor = np.matmul(dcostdx + agent.gamma * lmbda, B)
                    # print(f"TOTAL GRAD ACTOR = {grad_actor}")
                    # grad_actor = np.clip(grad_actor, -0.1, 0.1)
                    # grad_actor  = utils.overactuation_gradient_correction(gradients=grad_actor, actions=action, actions_clipped=action_clipped)
                    agent.update_actor(
                        x,
                        reference=R_sig,
                        gradient=grad_actor,
                    )
                    j += 1

                X_pred.append(x_next_pred)
                C_trained.append(cost.flatten())
                action_grad.append(grad_actor)
                critic_grad.append(grad_critic)
                list_F.append(A.flatten().copy())
                list_G.append(B.flatten().copy())
                list_RLS_cov.append(ac_model.cov.copy())

                ### Run environment ###
                action = agent.action(x, reference=R_sig)
                action = np.clip(
                    action,
                    np.array([np.radians(-25), 0.2]),
                    np.array([np.radians(25), 1.0]),
                )

                if total_steps < excitation_steps:
                    action += np.array(excitation_signal[total_steps])
                action = np.clip(
                    action,
                    np.array([np.radians(-25), 0.2]),
                    np.array([np.radians(25), 1.0]),
                )
                x_next, reward, _, _ = env.step(np.squeeze(action))
                total_steps += 1
                episode_steps += 1
                if episode_steps >= max_steps:
                    print("Max_step achived")
                    done = True

                model_error = ((x_next_pred - x_next) ** 2).mean()

                ### Real Cost ###
                e = np.matmul(P, (x_next.reshape([1, -1, 1]) - x_ref))
                cost = np.matmul(np.matmul(e.transpose(0, 2, 1), Q), e)

                R.append(reward)
                X.append(x_next)
                U.append(np.squeeze(action))
                e_model.append(model_error)
                C_real.append(cost)

                ### Update Model ###
                ac_model.update(x, action, x_next)

                ### Bookkeeping ###
                x = x_next
    return X, U, U_ref, critic_grad, action_grad, C_real, C_trained, total_steps, R


def plot_result(X, U, U_ref, critic_grad, action_grad, C_real, C_trained, R, name):
    _, axis = plt.subplots(4, 3, figsize=(14, 10))
    axis[0, 0].plot([i[0] for i in X], label="L[m]")
    axis[0, 0].plot([i[1] for i in X], label="H[m]")
    # axis[0, 0].plot([np.squeeze(i)[0] for i in X_pred], label="X position (pred)")
    axis[0, 0].plot([np.squeeze(i)[1] for i in U_ref], "--", label="H position (ref)")
    # axis[0, 0].plot([np.squeeze(i)[1] for i in X_pred], label="Y position (pred)")
    axis[0, 0].legend()

    axis[0, 1].plot([np.degrees(i[2]) for i in X], label="w_z [deg/s]")
    axis[0, 1].plot([np.degrees(i[3]) for i in X], label="theta [deg]")
    axis[0, 1].plot(
        [np.degrees(np.squeeze(i)[3]) for i in U_ref], "--", label="theta (ref)"
    )
    axis[0, 1].legend()

    axis[0, 2].plot([i[4] for i in X], label="V [m/s]")
    axis[0, 2].legend()

    axis[1, 2].plot([np.degrees(i[5]) for i in X], label="alpha [deg]")
    axis[1, 2].legend()

    axis[1, 1].plot([np.degrees(i[0]) for i in U], label="stab [deg]")
    axis[1, 0].legend()

    axis[1, 1].plot([i[1] for i in U], label="throttle [%]")
    axis[1, 1].legend()

    axis[2, 0].plot(np.squeeze(C_real), label="cost_real")
    axis[2, 0].plot(np.squeeze(C_trained), label="cost_predicted")
    axis[2, 0].legend()

    axis[2, 1].plot([np.squeeze(i) for i in critic_grad], label="critic grad")
    axis[2, 1].legend()
    axis[3, 1].plot([np.squeeze(i) for i in action_grad], label="actor grad")
    axis[3, 1].legend()

    axis[2, 2].plot(R, label="Reward")
    axis[2, 2].legend()

    plt.savefig(f"./logs/{name}")


def string_from_list(arr):
    string_value = " ".join(str(i).replace(".", "_") for i in arr)
    return string_value


def init_models(hyper_params, weights):
    TENSORBOARD_DIR = "./logs/tensorboard/DHP/"
    state_size = hyper_params.get("state_size")
    action_size = hyper_params.get("action_size")
    lr_critic = hyper_params.get("lr_critic")
    lr_actor = hyper_params.get("lr_actor")
    gamma_actor = hyper_params.get("gamma_actor")
    tracked = hyper_params.get("TRACKED")
    ac_kwargs = {
        # Arguments for all model types
        "state_size": state_size,
        "action_size": action_size,
        "predict_delta": False,
        # Neural Network specific args:
        "hidden_layer_size": [100, 100, 100],
        "activation": tf.nn.relu,
        # RLS specific args:
        "gamma": 0.9995,
        "covariance": 100,
        "constant": True,
        # LS specific args:
        "buffer_length": 10,
    }
    ac_model = model.RecursiveLeastSquares(**ac_kwargs)

    kwargs = {
        "input_size": [
            state_size,
            np.sum(tracked),
        ],  # [Aircraft state size, Number of tracked states]
        "output_size": action_size,  # Actor output size (Critic output is dependend only on aircraft state size)
        "hidden_layer_size": [
            50,
            50,
            50,
        ],  # List with number of nodes per layer, number of layers is variable
        "kernel_stddev": 0.1,  # Standard deviation used in the truncated normal distribution to initialize all parameters
        "lr_critic": lr_critic,  # Learn rate Critic
        "lr_actor": lr_actor,  # Learn rate Actor
        "gamma": gamma_actor,  # Discount factor
        "use_bias": False,  # Use bias terms (only if you train using batched data)
        "split": False,  # Split architechture of the actor, if False, a single fully connected layer is used.
        "target_network": False,  # Use target networks
        "activation": tf.keras.layers.Activation("relu"),
        "log_dir": TENSORBOARD_DIR,  # Where to save checkpoints
        "use_delta": (
            False,
            tracked,
        ),  # (True, TRACKED) used 's = [x, (x - x_ref)]' || (False, None) uses 's = [x, x_ref]'
    }
    agent = DHP.Agent(**kwargs)
    return agent, ac_model


def init_env():
    u_trimmed = np.array([np.radians(-4.3674), 0.3767])
    init_state = np.array([0, 3000, 0, np.radians(2.7970), 200, np.radians(2.7970)])
    env = Env(init_state, u_trimmed)
    state_size = len(init_state)
    action_size = 2
    TRACKED = [False, True, True, True, True, True]
    return state_size, action_size, TRACKED, env


def optimize_fun(weights):
    state_size, action_size, TRACKED, env = init_env()
    hyper_params = {
        "state_size": state_size,
        "action_size": action_size,
        "lr_critic": 0.1,
        "lr_actor": 0.05,
        "gamma_actor": 0.4,
        "TRACKED": TRACKED,
    }

    agent, ac_model = init_models(hyper_params, weights)
    (
        X,
        U,
        U_ref,
        critic_grad,
        action_grad,
        C_real,
        C_trained,
        total_steps,
        R,
    ) = run_train(env, agent, ac_model, weights, TRACKED)

    plot_name = RUN_TIME + string_from_list(weights) + ".png"
    plot_result(X, U, U_ref, critic_grad, action_grad, C_real, C_trained, R, plot_name)
    cost_step = sum(C_real) / total_steps
    return cost_step


weights = [1, 1, 2, 1, 1, 100]

cost = optimize_fun(weights)
