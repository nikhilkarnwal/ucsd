from torch import nn
from tqdm import tqdm
from actor_v2 import actor_policy_grad
import torch
import numpy as np
import matplotlib.pyplot as plt
import json


def update_policy(model, rollout, arg_dict):
    rollout = np.array(rollout)
    fin_r, prob = rollout[:, 0], np.log(rollout[:, 1])
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
    print(f'Grad Val {policy_gradient.item()}')
    policy_gradient.requires_grad = True
    if policy_gradient.item() == 0:
        return False
    model.train()
    model.optimizer.zero_grad()
    policy_gradient.backward()
    nn.utils.clip_grad_norm_(model.parameters(), arg_dict['grad_clip'])
    model.optimizer.step()
    return True


def policy_grad(model, arg_dict):
    pbar = tqdm(range(arg_dict['n_epi']), position=0, leave=True)
    win_all = []
    avg_win_all = []
    all_reward = []
    avg_all_reward = []
    score_all = []
    avg_score_all = []
    last_steps = 0
    total_steps = 0
    for n_epi in pbar:
        rollout, summary = actor_policy_grad(n_epi, model, arg_dict)
        check = update_policy(model, rollout, arg_dict)
        if not check:
            print(f'Gradient at Episode {n_epi} is 0')
        (win, score, tot_reward, steps, _, _, _, _) = summary
        pbar.set_postfix(Episode=n_epi, Win=win, Score=score, Reward=tot_reward, Steps=steps)
        win_all.append(win)
        avg_win_all.append(np.mean(win_all[-arg_dict['avg_steps']:]))
        all_reward.append(tot_reward)
        avg_all_reward.append(np.mean(all_reward[-arg_dict['avg_steps']:]))
        score_all.append(score)
        avg_score_all.append(np.mean(score_all[-arg_dict['avg_steps']:]))
        total_steps += steps
        last_steps = save_model(model, arg_dict, total_steps, last_steps)

    plt.figure(figsize=(7, 5), dpi=250)
    plt.plot(win_all, '^', ls='-', c='b', label='raw value')
    plt.plot(avg_win_all, 'o', ls='-', c='orange', label='smooth value')
    plt.xlabel('Episodes')
    plt.ylabel('WinRate')
    plt.legend()
    plt.savefig(arg_dict['dir'] + '/policy_win_rate', dpi=200)

    plt.figure(figsize=(7, 5), dpi=250)
    plt.plot(all_reward, '^', ls='-', c='b', label='raw value')
    plt.plot(avg_all_reward, 'o', ls='-', c='orange', label='smooth value')
    plt.xlabel('Episodes')
    plt.ylabel('RewardRate')
    plt.legend()
    plt.savefig(arg_dict['dir'] + '/policy_reward_rate', dpi=200)

    plt.figure(figsize=(7, 5), dpi=250)
    plt.plot(score_all, '^', ls='-', c='b', label='raw value')
    plt.plot(avg_score_all, 'o', ls='-', c='orange', label='smooth value')
    plt.xlabel('Episodes')
    plt.ylabel('ScoreRate')
    plt.legend()
    plt.savefig(arg_dict['dir'] + '/policy_score_rate', dpi=200)

    metrics = [win_all, avg_win_all, all_reward, avg_all_reward, score_all, avg_score_all]

    with open(arg_dict['dir'] + '/metrics.json', 'w') as file_ptr:
        json.dump(metrics, file_ptr)


def save_model(model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step:
        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        path = arg_dict["dir"] + "/model" + ".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return optimization_step
    else:
        return last_saved_step
