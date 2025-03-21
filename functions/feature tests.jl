#Here we tested different sets of features and generated graphs for these.

env = Environment(environment_type = :waterworld)
lps_simplest = LearningParameters(environment = env, feature_type = WWRadialFeatures(false, []))
lps_horizon_only = LearningParameters(environment = env, feature_type = WWRadialFeatures(true, []))
lps_radial = LearningParameters(environment = env, feature_type = WWRadialFeatures(true, [1]))
lps_radial2 = LearningParameters(environment = env, feature_type = WWRadialFeatures(true, [1,2]))

tr_simplest = mean_and_sterr(n_step_SARSA, 10, param = lps_simplest); 
tr_horizon_only = mean_and_sterr(n_step_SARSA, 10, param = lps_horizon_only);
tr_radial = mean_and_sterr(REINFORCE, 10, param = lps_radial);
tr_radial2 = mean_and_sterr(REINFORCE, 10, param = lps_radial2);

comp_means = plot(tr_simplest[1], xlabel = "", ylabel = "Avg. rolling returns", label = "Only avoid water")
plot!(tr_horizon_only[1], label = "Avoid water and look at horizon")
plot!(tr_radial[1], label = "Include 1 radial feature set")
plot!(tr_radial2[1], label = "Include 2 radial feature sets", legendposition = :right)

comp_stderr = plot(tr_simplest[2], xlabel = "Episode number", ylabel = "Std. err.", label = "")
plot!(tr_horizon_only[2], label = "")
plot!(tr_radial[2], label = "")
plot!(tr_radial2[2], label = "")

pl = plot(comp_means, comp_stderr, layout = @layout [a;b])
savefig("The impact of features")

r=1
R=200

comp_means = plot(tr_radial[1][r:R], xlabel = "", ylabel = "Avg. rolling returns", label = "Include 1 radial feature set")
plot!(tr_radial2[1][r:R], label = "Include 2 radial feature sets", legendposition = :right)

comp_stderr = plot(tr_radial[2][r:R], xlabel = "Episode number", ylabel = "Std. err.", label = "")
plot!(tr_radial2[2][r:R], label = "")

pl = plot(comp_means, comp_stderr, layout = @layout [a;b])
savefig("Small diff more feats")
