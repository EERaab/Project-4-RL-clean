
"""
    run_scenario(thetas, lps::LearningParameters, underlying_map::Matrix, pol_type::Symbol; episode_length = 100, explored = nothing, threshold = length(underlying_map))

Lets the agent run a scenario (no learning!) with the giving policy encoded by thetas and on the underlying map. 
"""
function run_scenario(thetas, lps::LearningParameters, underlying_map::Matrix, pol_type::Symbol; max_length = 100, explored = nothing)
    gz = size(underlying_map,1)
    
    env = Environment(gridsize = gz, underlying_map = underlying_map, environment_type = :waterworld)
    lps_adjusted = LearningParameters(environment = env, discount_factor = lps.discount_factor, epsilon = lps.epsilon, feature_type = lps.feature_type, qhat_type = lps.qhat_type)

    states = [initial_state(env)]
    if !(explored === nothing)
        states[end].map .= underlying_map .* explored
    end

    rewards = Float64[]
    actions = []
    elapsed = 1
    threshold = length(underlying_map)
    while elapsed ≤ max_length && any(states[end].map .== 0) && elapsed ≤ threshold

        if pol_type == :epsgreedy
            sa_feat = state_action_features(states[end], lps_adjusted)
            qh = qhat(sa_feat, thetas, lps_adjusted.qhat_type)
            push!(actions, greedy_action(qh, 0; eps_greedy = true, param = lps_adjusted))
        elseif pol_type == :greedy
            sa_feat = state_action_features(states[end], lps_adjusted)
            qh = qhat(sa_feat, thetas, lps_adjusted.qhat_type)
            push!(actions, greedy_action(qh, 0; eps_greedy = false, param = lps_adjusted))
        elseif pol_type == :softmax
            sa_feat = state_action_features(states[end], lps_adjusted)
            pol = soft_max_vector(thetas, sa_feat)
            push!(actions, action(pol))
        else
            error("Improper policy type")
        end
        state, reward = transition_and_reward!(states[end], actions[end], env, in_place = false) 
        push!(states, state)
        push!(rewards, reward)
        elapsed += 1 
    end
    return (states, rewards, actions)
end

"""
    create_random_bool_matrix(gz, density)

Creates a random bool matrix.
"""
function create_random_bool_matrix(gz, density)
    n = gz^2    
    num_ones = round(Int, n * density)
    matrix_elements = vcat(ones(Int, num_ones), zeros(Int, n - num_ones))
    shuffle!(matrix_elements)    
    return reshape(matrix_elements, gz, gz)
end

"""
    returns_scenario(t::Tuple, g::Float64)

Computes the returns for a scenario.
"""
function returns_scenario(t::Tuple, g::Float64)
    return dot(t[2], g .^(0:(length(t[2])-1)))
end

"""
    returns_scenario(t::Tuple, lps::LearningParameters)

Computes the returns for a scenario.
"""
function returns_scenario(t::Tuple, lps::LearningParameters)
    return dot(t[2], lps.discount_factor .^(0:(length(t[2])-1)))
end

"""
    returns_scenario(rewards::Vector, g::Float64)

Computes the returns for a given vector of rewards.
"""
function returns_scenario(rewards::Vector, g::Float64)
    return dot(rewards, g .^(0:(length(actions)-1)))
end

"""
    returns_scenario(actions::Vector, lps::LearningParameters)

Computes the returns for a given vector of rewards.
"""
function returns_scenario(actions::Vector, lps::LearningParameters)
    return dot(rewards, lps.discount_factor .^(0:(length(actions)-1)))
end