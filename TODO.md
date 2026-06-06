## Plotting
1. Normalize the scale across runs. When we plot the same x/y heatmap we ideally want the scale normalized across runs, e.g alpha vs sampling training loss we probably want the same scale for the sampler=uniform and sampler=bandit so that we can compare them

2. Add 3D heatmaps in the current plotting config. We have x and y but it's also possible to add a z axis


## Other

1. The reward metric is a bit flawed: it is clearly proportional to the number of sampled nodes as it is nonnegative. So for example when we study sampling ratio vs reward we're not gonna have any meaningful results because the reward is just going to increase. The real reward metric we want to observe is the reward averaged over the cardinality of the sampled neighbors. e.g if we are sampling 9 neighbors we divide the round's reward by 9. Same should be for the oracle reward. The regret is going to be naturally scaled as well once we do this as it's defined as oracle reward - self reward 

2. It would be interesting to also keep a max and min reward per-node-per-round. So we can observe how the highest/lowest reward a node gets evolves over rounds. It could be added as a subplot in the reward plot, where we have the per-node average/max/min/median of the per-round max and min reward. It's basically as the other node-aggregated metrics.
