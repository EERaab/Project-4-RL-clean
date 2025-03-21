### Here we analysed the balancing matrix (represented as a tensor)

env = Environment(environment_type = :open)
lps_fwd_only = LearningParameters(environment = env, feature_type = RadialFeatures(true,[1]), balancing_tensor = balancing_tensor(bwd = 0.0))
lps_fwd_and_bwd = LearningParameters(environment = env, feature_type = RadialFeatures(true,[1]))

fwd = mean_and_sterr(n_step_SARSA, 10, param = lps_fwd_only, rollnumber = 30)

fwd_and_bwd = mean_and_sterr(n_step_SARSA, 10, param = lps_fwd_and_bwd, rollnumber = 30)

comp_means = plot(fwd[1], xlabel = "", ylabel = "Avg. rolling returns", label = "Only forward measure")
plot!(fwd_and_bwd[1], label = "Forward and backward measure", legendposition = :bottomright)

comp_stderr = plot(fwd[2], xlabel = "Episode number", ylabel = "Std. err.", label = "")
plot!(fwd_and_bwd[2], label = "")

pl = plot(comp_means, comp_stderr, layout = @layout [a;b])
savefig("Comparing balancing tensor structures")