# Some plotting functions that were used in analyzing many runs.

"""
    mean_and_sterr2(F::Function, param::LearningParameters; runs::Int = 10, rollnumber = 10)

Returns the same means and standard errors as 'mean_and_sterr' but stores the raw run data. Memory-intesive obviously.
"""
function mean_and_sterr2(F::Function, param::LearningParameters; runs::Int = 10, rollnumber = 10)
    methodfunction = x -> F(param = param, plotting = true) 
    #The function F is our method (i.e. n_step_SARSA or similar. It is assumed to output [parameter vectors, returns,---]
    Outcome = methodfunction(1)
    z = rollmean(Outcome[2], rollnumber) 
    df = Array{Float64}(undef, length(z), runs)
    raw_data = [Outcome] #oough
    df[:,1] = z
    for j = 2:runs
        Outcome = methodfunction(1)
        df[:,j] = rollmean(Outcome[2], rollnumber) 
        push!(raw_data, Outcome)
    end
    means = Float64[]
    stderrors = Float64[]
    for j ∈ 1:size(df)[1]
        push!(means, mean(df[j,:]))
        push!(stderrors, std(df[j,:]))
    end
    stderrors *= 1/sqrt(runs);
    return means, stderrors, raw_data
end

"""
    plotrun2(func::Function, param::LearningParameters; rollnumber = 10, runs::Integer = 10)

Returns the same plots as 'plotrun' but stores the raw run data. Memory-intesive obviously.
"""
function plotrun2(func::Function, param::LearningParameters; rollnumber = 10, runs::Integer = 10)
    run = mean_and_sterr2(func, param, runs = runs, rollnumber = 10)
    pl1 = plot(run[1], ylabel = "Rolling avg.", label = "", minoraxis = true)
    pl2 = plot(run[2], ylabel = "Std.err.", xlabel = "Episode number", label = "", minoraxis = true)
    print("RL-scheme: ")
    println(func)
    print("Number of runs: ")
    println(runs)
    print("Rolling average over: ")
    print(rollnumber)
    println(" episodes")
    print("Environment gridsize: ")
    println(param.environment.gridsize)
    print("Environment mapping radius: ")
    println(param.environment.mapping_radius)
    print("Discount factor (gamma): ")
    println(param.discount_factor)
    if func == n_step_SARSA
        print("Epsilon: ")
        println(param.epsilon)
        print("Epsilon is decaying: ")
        println(param.epsilon_decaying)
        print("Episode max. length: ")
        println(param.episode_length)
        print("Episode number: ")
        println(param.episode_number)
        print("Learning rate: ")
        println(param.learning_rate)
        print("Step number in n-step SARSA: ")
        println(param.step_number)
    end
    print("Feature type (see code or ask Erik for definition): ")
    println(typeof(param.feature_type))
    return plot(pl1,pl2,layout = @layout [a; b]), run[3]
end