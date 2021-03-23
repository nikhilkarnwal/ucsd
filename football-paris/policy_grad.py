from torch import nn
from tqdm import tqdm
from actor_v2 import actor_policy_grad
import torch
import numpy as np
import matplotlib.pyplot as plt


def update_policy(model, rollout, arg_dict):
    fin_r, prob = rollout[:, 0], rollout[:, 1]
    disc_r = []
    GAMMA = arg_dict['gamma']
    for t in range(len(fin_r)):
        Gt = 0
        pw = 0
        for r in fin_r[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        disc_r.append(Gt)

    disc_r = torch.tensor(disc_r)
    disc_r = (disc_r - disc_r.mean()) / (disc_r.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(prob, disc_r):
        policy_gradient.append(-log_prob * Gt)
    policy_gradient = torch.stack(policy_gradient).sum()
    model.optimizer.zero_grad()
    policy_gradient.backward()
    nn.utils.clip_grad_norm_(model.parameters(), arg_dict['grad_clip'])
    model.optimizer.step()


def policy_grad(model, arg_dict):
    pbar = tqdm(range(arg_dict['n_epi']))
    num_steps = []
    avg_num_steps = []
    all_reward = []
    last_steps = 0
    for n_epi in pbar:
        rollout, summary = actor_policy_grad(model, arg_dict)
        update_policy(model, rollout, arg_dict)
        (win, score, tot_reward, steps, _, _, _, _) = summary
        pbar.set_postfix(Episode=n_epi, Win=win, Score=score, Reward=tot_reward, Steps=steps)
        num_steps.append(steps)
        avg_num_steps.append(np.mean(num_steps[-arg_dict['avg_steps']:]))
        all_reward.append(tot_reward)
        last_steps = save_model(model, arg_dict, np.sum(num_steps), last_steps)

    plt.plot(num_steps)
    plt.plot(avg_num_steps)
    plt.xlabel('Episodes')
    plt.show()
    plt.savefig(arg_dict['dir'] + '/policy_graph', dpi=200)


def save_model(model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step + arg_dict["model_save_interval"]:
        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        path = arg_dict["dir"] + "/model_" + str(optimization_step) + ".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return optimization_step
    else:
        return last_saved_step
