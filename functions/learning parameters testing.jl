### Basic grid search and plotting for simple parameters.
### Here we see that we should've used a different default epsilon
### Learning rates were well chosen

env = Environment(environment_type = :waterworld)


mid_epislon = LearningParameters(environment = env, epsilon = 0.1, epsilon_decaying = true)
low_epsilon = LearningParameters(environment = env, epsilon = 0.01, epsilon_decaying = true)
ultra_low_epsilon = LearningParameters(environment = env, epsilon = 0.001, epsilon_decaying = true)

tr_mid = mean_and_sterr(n_step_SARSA, 10, param = mid_epislon);
tr_low = mean_and_sterr(n_step_SARSA, 10, param = low_epsilon);
tr_ultra_low = mean_and_sterr(n_step_SARSA, 10, param = ultra_low_epsilon);

r=1
R=800

comp_means = plot(tr_mid[1][r:R], xlabel = "", ylabel = "Avg. rolling returns", label = L"$\epsilon = 0.1$")
plot!(tr_low[1][r:R], label = L"$\epsilon = 0.01$")
plot!(tr_ultra_low[1][r:R], label = L"$\epsilon = 0.001$", legendposition = :right)

comp_stderr = plot(tr_mid[2][r:R], xlabel = "Episode number", ylabel = "Std. err.", label = "")
plot!(tr_low[2][r:R], label = "")
plot!(tr_ultra_low[2][r:R], label = "")

pl = plot(comp_means, comp_stderr, layout = @layout [a;b])
savefig("We should have set epsilon to lower")

mid_learning = LearningParameters(environment = env, epsilon = 0.1, epsilon_decaying = true, learning_rate = 0.1)
low_learning = LearningParameters(environment = env, epsilon = 0.1, epsilon_decaying = true, learning_rate = 0.01)
ultra_low_learning = LearningParameters(environment = env, epsilon = 0.1, epsilon_decaying = true, learning_rate = 0.001)


tr_mid_learn = mean_and_sterr(n_step_SARSA, 10, param = mid_learning);
tr_low_learn = mean_and_sterr(n_step_SARSA, 10, param = low_learning);
tr_ultra_low_learn = mean_and_sterr(n_step_SARSA, 10, param = ultra_low_learning);


tr_mid_learn_RF = mean_and_sterr(REINFORCE, 10, param = mid_learning);
tr_low_learn_RF = mean_and_sterr(REINFORCE, 10, param = low_learning);
tr_ultra_low_learn_RF = mean_and_sterr(REINFORCE, 10, param = ultra_low_learning);


r=1
R=100

comp_means = plot(tr_mid_learn[1][r:R], xlabel = "", ylabel = "Avg. rolling returns", label = L"$\alpha = 0.1$")
plot!(tr_low_learn[1][r:R], label = L"$\alpha = 0.01$")
plot!(tr_ultra_low_learn[1][r:R], label = L"$\alpha = 0.001$", legendposition = :right)

comp_stderr = plot(tr_mid_learn[2][r:R], xlabel = "Episode number", ylabel = "Std. err.", label = "")
plot!(tr_low_learn[2][r:R], label = "")
plot!(tr_ultra_low_learn[2][r:R], label = "")

pl = plot(comp_means, comp_stderr, layout = @layout [a;b])


r=1
R=800

comp_means = plot(tr_mid_learn_RF[1][r:R], xlabel = "", ylabel = "Avg. rolling returns", label = L"$\alpha = 0.1$")
plot!(tr_low_learn_RF[1][r:R], label = L"$\alpha = 0.01$")
plot!(tr_ultra_low_learn_RF[1][r:R], label = L"$\alpha = 0.001$", legendposition = :right)

comp_stderr = plot(tr_mid_learn_RF[2][r:R], xlabel = "Episode number", ylabel = "Std. err.", label = "")
plot!(tr_low_learn_RF[2][r:R], label = "")
plot!(tr_ultra_low_learn_RF[2][r:R], label = "")

pl = plot(comp_means, comp_stderr, layout = @layout [a;b])

