#This is the main file for REINFORCE

"""
    action(policy_vec::Vector{Float64})

Generates an action from a policy vector.
"""
function action(policy_vec::Vector{Float64})
    p = rand()
    #goofy implementation but since we only have 4 actions...
    if p < policy_vec[1]
        return 1
    elseif p < policy_vec[2]+policy_vec[1]
        return 2
    elseif p < policy_vec[3]+policy_vec[2]+policy_vec[1]
        return 3
    else
        return 4
    end
end

"""
    baseline_parameter_update!(G, omegas, state_action_features, lps::LearningParameters)

Currently unfinished, assuming no baseline here.
"""
function baseline_parameter_update!(G, omegas, state_action_features, lps::LearningParameters)
    return 1.0 #right now this does nothing, but if we introduce a baseline this must be adjusted.
end

"""
    parameter_update_REINFORCE!(iter::Integer, δ::Float64, policy_vec::Vector{Float64}, state_action_feats::Vector{Matrix{Float64}}, action::Integer, thetas, lps::LearningParameters)

Takes a parameter vector (of a linear-exponent softmax policy) and updates it according to REINFORCE update rules.

Only works for open world, implementation is different for waterworld.
"""
function parameter_update_REINFORCE!(iter::Integer, δ::Float64, policy_vec::Vector{Float64}, state_action_feats::Vector{Matrix{Float64}}, action::Integer, thetas, lps::LearningParameters)
    mult = δ*lps.learning_rate*(lps.discount_factor^iter)
    for j ∈ eachindex(thetas)
       thetas[j] .+= mult*gradient_ln_soft_max(policy_vec, action, state_action_feats, opt = j) 
    end
end

function REINFORCE(;param = LearningParameters(), plotting::Bool = false, thetas = nothing)
    
    if thetas === nothing
        thetas = initial_parameter_vector_REINFORCE(param)
    end
    discounted_returns = Float64[]
    
    #should probably have the same structure for REINFORCE
    omegas = initial_baseline_parameter_vector(param)


    if plotting
        stateset = []
        rewardset = []
        policyset = []
        explored = []
    end

    for i ∈ 1:param.episode_number
        if param.environment.environment_type == :waterworld && i>1
            new_map!(param.environment) #could cause visualization bugs
        end
        active_state = initial_state(param.environment)
        if plotting
            states = [active_state]
        end
        sa_feats = [state_action_features(active_state, param)]
        policies = [soft_max_vector(thetas, sa_feats[end])] #here we should really have a more general structure. This just assumes we always use the soft_max_vector
        actions = [action(policies[end])]
        rewards = Float64[]

        l = 1
        #The code below generates a full episode, its pretty goofy.
        while l ≤ param.episode_length | !any(iszero.(active_state.map))
            if plotting
                state, reward = transition_and_reward!(states[end], actions[end], param.environment, in_place = false)
                push!(states, state)
                active_state = states[end] #this should work
            else
                reward = transition_and_reward!(active_state, actions[end], param.environment)
            end
            push!(rewards, reward)


            if l == param.episode_length | !any(iszero.(active_state.map))
                break
            else
                l+=1 
                push!(sa_feats, state_action_features(active_state, param))
                push!(policies, soft_max_vector(thetas, sa_feats[end]))
                push!(actions, action(policies[end]))
            end
        end 

        G = 0.0
        for t = 0:(l-1)
            G = dot(param.discount_factor .^(0:l-t-1), rewards[t+1:end])
            δ = G
            #Unimplemented baseline. When implemented, use this line:
            #δ = baseline_parameter_update!(G, omegas, state_action_features, param) 
            parameter_update_REINFORCE!(t, δ, policies[t+1], sa_feats[t+1], actions[t+1], thetas, param) 
        end    

        push!(discounted_returns, dot(param.discount_factor.^(0:l-1), rewards))
        if plotting
            push!(stateset, states)
            push!(rewardset, rewards)
            push!(policyset, policies)
            push!(explored, count(.!iszero.(states[end].map)))
        end
    end    
    if plotting
        gz = Float64(param.environment.gridsize)^2
        return thetas, discounted_returns, (stateset, rewardset, explored/gz, policyset)
    else
        return thetas, discounted_returns
    end
end