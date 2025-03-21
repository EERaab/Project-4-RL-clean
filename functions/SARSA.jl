#This is the main SARSA method file.

"""
    random_maximizer(arr::Array)::Integer

Returns the index of a random maximal element in an array. Used in greedy actions where we need to tie-break.
"""
function random_maximizer(arr::Array)::Integer
    M = arr[1]
    ind = Int8[1]
    for i ∈ eachindex(arr)
        if arr[i] > M
            ind = Int8[i]
            M = arr[i]
        elseif arr[i] == M
            push!(ind,i)
        end
    end
    return rand(ind)
end

"""
    greedy_action(qhat::Array{Float64}, iter::Integer; eps_greedy::Bool = true, param = Parameters())

Takes a vector qhat and returns a random qhat maximal action with probability 1 - param.epsilon (or some decayed version thereof), and otherwise just a random action.
"""
function greedy_action(qhat::Array{Float64}, iter::Integer; eps_greedy::Bool = true, param = Parameters())
    if eps_greedy
        if param.epsilon_decaying 
            if param.decay_function(param.epsilon, iter) > rand()
                return rand(Int8(1):Int8(4))
            end
        elseif param.epsilon > rand()
            return rand(Int8(1):Int8(4))
        end
    end
    return random_maximizer(qhat)
end

"""
    update_thetas_n_step_SARSA!(G::Float64, sa_feats::Vector{Matrix{Float64}}, thetas, qhat::Vector{Float64}, action::Integer, param::LearningParameters)

Takes the parameter vector theta and updates it according to semi-gradient n_step_SARSA rules using state-action featuers.
"""
function update_thetas_n_step_SARSA!(G::Float64, sa_feats::Vector{Matrix{Float64}}, thetas, qhat::Vector{Float64}, action::Integer, param::LearningParameters)
    for j ∈ eachindex(thetas)
        thetas[j] += param.learning_rate*(G-qhat[action])*qhat_grad(sa_feats, thetas, action, param.qhat_type, opt = j)
    end
end

"""
    update_thetas_n_step_SARSA!(G::Float64, sa_feats::StateActionFeaturesWaterworld, thetas::WaterworldParameters, qhat::Vector{Float64}, action::Integer, param::LearningParameters)

The parameter update function SARSA applied to Waterworld.
"""
function update_thetas_n_step_SARSA!(G::Float64, sa_feats::StateActionFeaturesWaterworld, thetas::WaterworldParameters, qhat::Vector{Float64}, action::Integer, param::LearningParameters)
    mult = param.learning_rate*(G-qhat[action])
    thetas.water_blocking_param += mult*sa_feats.water_blocking[action]
    for j ∈ eachindex(thetas.beyond_params) #basically looping over different distances here
        thetas.beyond_params[j] .+= mult*sa_feats.beyond[j][:,action]
    end
    for j ∈ eachindex(thetas.unmapped_params) #basically looping over different distances here
        thetas.unmapped_params[j] .+= mult*sa_feats.unmapped[j][:,action]
    end
    for j ∈ eachindex(thetas.water_params) #basically looping over different distances here
        thetas.water_params[j] .+= mult*sa_feats.water[j][:,action]
    end
end

"""
    n_step_SARSA(;param = LearningParameters(), plotting::Bool = false)

Trains an agent using n_step_SARSA with parameters specified in param. If plotting = true then it will return every set of states and rewards as well.
"""
function n_step_SARSA(;param = LearningParameters(), plotting::Bool = false, thetas = nothing)
    discounted_returns = Float64[]
    if thetas === nothing
        thetas = initial_parameter_vector_SARSA(param)
    end
    if plotting
        stateset = []
        rewardset = []
        explored = []
    end
    
    for i ∈ 1:param.episode_number
        #this should be part of the initialization

        if param.environment.environment_type == :waterworld && i>1
            new_map!(param.environment) #could cause visualization bugs
        end
        states = [initial_state(param.environment)]
        sa_feats = [state_action_features(states[end], param)]
        qhats = [qhat(sa_feats[end], thetas, param.qhat_type)]
        actions = [greedy_action(qhats[end], i; param = param)]
        rewards = Float64[]

        t = 0
        T = param.episode_length 
        while true 
            if t < T 
                newS, reward = transition_and_reward!(states[t+1], actions[t+1], param.environment, in_place = false) #newstate uppdateras    
                push!(states, newS)
                push!(rewards, reward)
                if (t+1 == param.episode_length)|!any(iszero.(newS.map))
                    T = t + 1
                else
                    push!(sa_feats, state_action_features(states[end], param))
                    push!(qhats, qhat(sa_feats[end], thetas, param.qhat_type))
                    push!(actions, greedy_action(qhats[end], i; param = param))
                end
            end

            τ = t - param.step_number + 1

            if τ ≥ 0
                G = 0.0
                for i ∈ (τ+1):min(τ+param.step_number,T)
                    G += param.discount_factor^(i-τ-1)*rewards[i]
                end
                if τ + param.step_number < T 
                    G += (param.discount_factor^param.step_number)*qhats[τ+1+param.step_number][actions[τ+1+param.step_number]] 
                end
                update_thetas_n_step_SARSA!(G, sa_feats[τ+1], thetas, qhats[τ+1], actions[τ+1], param)
            end
            t += 1
            if (τ == T-1)
                break
            end
        end
        push!(discounted_returns, dot(param.discount_factor.^(0:T-1), rewards))
        if plotting
            push!(stateset, states)
            push!(rewardset, rewards)
            push!(explored, count(.!iszero.(states[end].map)))
        end
    end
    if plotting
        gz = Float64(param.environment.gridsize)^2
        return thetas, discounted_returns, (stateset, rewardset, explored/gz)
    else
        return thetas, discounted_returns
    end
end