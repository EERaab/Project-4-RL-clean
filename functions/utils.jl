#this is mostly plotting utilities and similar things

"""
    balancing_tensor(fwd = 1, ort = 1, bwd =1, ignore_orthogonal_directions::Bool = true)

In the linear balanced q-hat (described in Overleaf doc.) we have an action-dependent matrix B(a) (which sits "inside" a bigger matrix A(a)) which defines q-hat. This function outputs this matrix as a tensor R with B(a) = R[:,:,a].

**Note: It is quite inefficient to use this matrix as it is sufficiently sparse that a fixed addition instead of a bunch of matrix mult. may do _far_ better!** Version 2 is probably better, but slightly less flexible.
"""
function balancing_tensor(;fwd = 1.0, ort = 0.5, bwd = 1.0, ignore_orthogonal_directions::Bool = true)
    if !ignore_orthogonal_directions
        B = Array{Float64}(undef,3,4,4)
        B[:,:,1] = [fwd 0 0 0 ; 0 ort 0 ort ; 0 0 bwd 0]
        B[:,:,2] = [0 fwd 0 0 ; ort 0 ort 0 ; 0 0 0 bwd]
        B[:,:,3] = [0 0 fwd 0 ; 0 ort 0 ort ; bwd 0 0 0]
        B[:,:,4] = [0 0 0 fwd ; ort 0 ort 0 ; 0 bwd 0 0]
        return B
    else
        B = Array{Float64}(undef,2,4,4)
        B[:,:,1] = [fwd 0 0 0 ; 0 0 bwd 0]
        B[:,:,2] = [0 fwd 0 0 ; 0 0 0 bwd]
        B[:,:,3] = [0 0 fwd 0 ; bwd 0 0 0]
        B[:,:,4] = [0 0 0 fwd ; 0 bwd 0 0]
        return B
    end
end

"""
    mean_and_sterr(F::Function, runs::Int; rollnumber = 50, rolling = true, param = LearningParameters())

Returns the mean and standard error in the returns of a method F (which is n_step_SARSA or REINFORCE), for the given number of runs, using the parameters in 'param', and defaults to returning rolling averages over 50 episodes.
"""
function mean_and_sterr(F::Function, runs::Int; rollnumber = 50, rolling = true, param = LearningParameters())
    methodfunction = x-> F(param = param) 
    #The function F is our method (i.e. n_step_SARSA or similar. It is assumed to output [parameter vectors, returns]
    if rolling
        Outcome = methodfunction(1)
        z = rollmean(Outcome[2], rollnumber) 
        df = Array{Float64}(undef, length(z), runs)
        df[:,1] = z
        for j = 2:runs
            Outcome = methodfunction(1)
            df[:,j] = rollmean(Outcome[2], rollnumber) 
        end
    else
        df = Array{Float64}(undef, param.episode_number, runs)
        for j ∈ 1:runs
            Outcome = methodfunction(1)
            df[:,j] = Outcome[2]
        end
    end
    means = Float64[]
    stderrors = Float64[]
    for j ∈ 1:size(df)[1]
        push!(means, mean(df[j,:]))
        push!(stderrors, std(df[j,:]))
    end
    stderrors *= 1/sqrt(runs);
    return means, stderrors
end

"""
    plotrun(func::Function, runs::Integer, param; rollnumber = 50)

Plots the means and standard error from mean_and_sterr and prints relevant parameters.
"""
function plotrun(func::Function, param; rollnumber = 50, runs::Integer = 10)
    run = mean_and_sterr(func, runs, rollnumber = rollnumber, param = param)
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
    if func == n_step_SARSA
        print("Step number in n-step SARSA: ")
        println(param.step_number)
    end
    print("Feature type (see code or ask Erik for definition): ")
    println(typeof(param.feature_type))
    plot(pl1,pl2,layout = @layout [a; b])
end

"""
    plotstate(state; title = "", dry_state::Bool = false)

Plots the state, i.e. the map with the explored tiles and the agent position.
"""
function plotstate(state; title = "", dry_state::Bool = false)
    if dry_state
        c = cgrad([:yellow, :black, :green])
    else
        c = cgrad([:yellow, :black, :green, :blue])
    end
    mwa  = copy.(state.map)
    mwa[state.agent_position...] += -2
    return heatmap(mwa, xaxis=false, yaxis=false, colorbar = false, yflip = true, title = title, color = c)
end

"""
    create_animation(stateset, rewardset; name::String = "Exploradora", steps_from_end = 0)

Creates a gif showing the path and rewards taken by an agent.
"""
function create_animation(stateset, rewardset; name::String = "Exploradora", steps_from_end = 0)
    dry = !any(stateset[end-steps_from_end][end].map .== 2)
    plots = []
    anim = @animate for i=1:length(stateset[end-steps_from_end])
        state = stateset[end-steps_from_end][i]
        if i == 1
            push!(plots,plotstate(state, title = "Initial state", dry_state = dry))
        elseif i<length(stateset[end-steps_from_end])
            push!(plots, plotstate(state, title = "Reward: "*string(rewardset[end-steps_from_end][i-1]), dry_state = dry))
        else
            push!(plots, plotstate(state, title = "Reward: "*string(rewardset[end-steps_from_end][i-1])*", terminal state.", dry_state = dry))
        end
    end
    gif(anim, name *".gif", fps = 4)
end


"""
    create_animation(t::Tuple; name::String = "Exploradora", steps_from_end = 0)

Creates a gif showing the path and rewards taken by an agent, and accepts the tuples that are output by the n_step_SARSA and REINFORCE functions.
"""
function create_animation(t::Tuple; name::String = "Exploradora", steps_from_end = 0)
    create_animation(t[3][1],t[3][2], name = name, steps_from_end = steps_from_end)
end

"""
    create_animation_scenario(scen; name::String = "scenario", steps_from_end = 0)

Animation for scenarios.
"""
function create_animation_scenario(scen; name::String = "scenario", steps_from_end = 0)
    plots = []
    dry = false
    anim = @animate for i=1:length(scen[1])
        state = scen[1][i]
        if i == 1
            push!(plots,plotstate(state, title = "Initial state", dry_state = dry))
        elseif i<length(scen[1])
            push!(plots, plotstate(state, title = "Reward: "*string(scen[2][i-1]), dry_state = dry))
        else
            push!(plots, plotstate(state, title = "Reward: "*string(scen[2][i-1])*", terminal state.", dry_state = dry))
        end
    end
    gif(anim, name *".gif", fps = 4)
end

"""
    create_animation(t::Tuple, index::Integer; name::String = "Exploradora")

Creates a gif showing the path and rewards taken by an agent, and accepts the tuples that are output by the n_step_SARSA and REINFORCE functions. Plots the training session at the index.
"""
function create_animation(t::Tuple, index::Integer; name::String = "Exploradora")
    create_animation(t[3][1],t[3][2], name = name, steps_from_end = length(t[3][1])-index)
end