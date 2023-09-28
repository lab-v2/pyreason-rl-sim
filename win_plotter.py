import matplotlib
import matplotlib.pyplot as plt
import json
  
f = open ('move_success_eval_1.json', "r")
metrics = json.loads(f.read())

def plot_scores():
    ## select only the 100 policy trials
    steps = sorted([int(x) for x in metrics.keys() if int(x) % 16000 == 0])
    epsticks = [f"{x//1000}K" for idx, x in enumerate(steps) if (idx+1) % 10 == 0]
    scores_t = [metrics[str(x)]["avg_rew"] for x in steps]
    ## All the eval trials run for 500 episodes, hence the division by 5 to get %
    win_pers = [metrics[str(x)]["win_per"]/5 for x in steps]

    x_ax = steps
    fig,ax = plt.subplots()
    win_ax = ax.twinx()
    ax.plot(x_ax,
            scores_t,
            color="blue", label='Avg. Reward')
    ax.set_xlabel("Steps (in thousands (K))", fontsize = 14)
    ax.set_ylabel("Avg. Reward",
                  color="blue",
                  fontsize=14)
    win_ax.plot(x_ax,
            win_pers,
            color="red")
    win_ax.set(ylim=[0, 100])
    win_ax.set_ylabel("Win Percentage",
                  color="red",
                  fontsize=14)

    # Create a legend, which with the twin axes requires spoofing diversity.
    ax.plot([], [], color="red", label='Win Percentage')
    ax.legend(loc='upper left')

#     ax.set_ylim(0, 105)
#     ax.set_xlim(995, 15005)
    ax.set_xticklabels(epsticks)
    ax.set_title("DQN Single-Agent Movement")
    # plt.show()
    fig.savefig('dqn_move_win_single.jpg',
                format='jpeg',
                dpi=500,
                bbox_inches='tight')

plot_scores()
