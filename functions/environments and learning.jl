# Here we encode basic structures for learning and environment

"""
    map_generator(gridsize::Integer, environment_type::Symbol; seed = rand(Int64))::Matrix{Int8}

Returns the underlying map of the world.
"""
function map_generator(gridsize::Integer, environment_type::Symbol; seed = rand(1:100000), density::Float64 = 0.23, starting_rad = 3)::Matrix{Int8}
    if environment_type == :open
        return ones(Int8, gridsize, gridsize)
    elseif environment_type == :waterworld
        return generate_connected_matrix(gridsize = gridsize, density = density, starting_rad = starting_rad)
        #return 1 .+ Int8.(py"generate_gridmap($gridsize, $seed)") #run for Python maps which are pretty but slow to generate
    else
        error("Please select implemented world-type.")
    end
end

@kwdef struct Environment
    gridsize::Int8 = Int8(51)
    mapping_radius::Int8 = 2
    mapping_reward::Float64  = 1.0
    re_mapping_penalty::Float64 = -1.0
    bumping_penalty::Float64 = 0.0
    environment_type::Symbol = :open #ugly
    water_density::Float64 = 0.1
    underlying_map::Matrix{Int8} = map_generator(gridsize, environment_type, density = water_density, starting_rad = mapping_radius+1)
end

"""
    new_map!(env::Environment)

Generates a new map for the environment!
"""
function new_map!(env::Environment)
    env.underlying_map .= generate_connected_matrix(gridsize = env.gridsize, density = env.water_density, starting_rad = env.mapping_radius+1)
end

"""
    default_decay(s::Float64, n::Int; threshold::Int = 0, hard_threshold::Bool = false)::Float64

The decay function for a decaying epsilon. 
"""
function default_decay(s::Float64, n::Int; threshold::Int = 0, hard_threshold::Bool = false)::Float64
    if n > threshold       
        if hard_threshold 
            return s*(0.99)^(n)
        else
            return s*(0.99)^(n-threshold)
        end            
    end
    return s    
end

#This structure is a vertically integrated set of parameters, which is to say, *anyting* that appears in any method. So e.g. policy type does nothing in SARSA, but is used in REINFORCE.
@kwdef struct LearningParameters
    environment::Environment = Environment()
    discount_factor::Float64 = 0.9
    epsilon::Float64 = 0.1
    epsilon_decaying::Bool = true
    decay_function::Function = (s,n) -> default_decay(s,n)
    episode_length::Int = 80
    episode_number::Int = 1000
    learning_rate::Float64 = 0.001
    step_number::Int = 4
    feature_type::FeatureType = default_features(environment)
    qhat_type::QhatType = LinearBalancedQhat() #used in SARSA, tells us how to compute q-hat. Leaves room for other implementations
    #policy_type::PolicyType = SoftMax()
    ignore_orthogonal_directions::Bool = true
    balancing_tensor::Array{Float64} = balancing_tensor(ignore_orthogonal_directions = ignore_orthogonal_directions)
    uses_balanced_features::Bool = true #So for certain methods it would be appropriate to use non-balanced features. This is not the case for us, but this makes sense to leave in here to allow for some flexibility later.
    baseline_learning_rate::Float64 = 0.001
    baseline_constant::Float64 = 0.0
    baseline_estimator::Symbol = :nil
end

"""
    initial_state(env::Environment)::State

Initializes agent in the middle of gridworld and with the correct tiles explored.
"""
function initial_state(env::Environment)::State
    position = fill(Int8(ceil(env.gridsize/2)),2)
    map = zeros(Int8, size(env.underlying_map))
    for i ∈ (-env.mapping_radius):(env.mapping_radius)
        for j ∈ (abs(i) - env.mapping_radius) : (env.mapping_radius - abs(i))
            map[position[1]+i, position[2]+j] = env.underlying_map[position[1]+i, position[2]+j]
        end
    end
    return State(position, map)
end

"""
    inbounds(position::Vector{<:Real}, action::Integer, env::Environment)::Bool

Checks whether agent would leave gridworld by taking the given action.
"""
function inbounds(position::Vector{<:Real}, action::Integer, env::Environment)::Bool
    if action == 1 && position[1] == 1
        return false
    elseif action == 2 && position[2] == env.gridsize
        return false
    elseif action == 3 && position[1] == env.gridsize
        return false
    elseif action == 4 && position[2] == 1
        return false
    end
    return true
end

"""
    inbounds(coordinates::Vector, env::Environment)::Bool

Checks whether the vector "coordinates" is in gridworld.
"""
function inbounds(coordinates::Vector, env::Environment)::Bool
    #this is slow and should generally be avoided as much as possible.
    return all(1 .≤ coordinates .≤ env.gridsize)
end

"""
    inbounds(coordinates::Vector, env::Environment)::Bool

Checks whether the coordinate is in the gridworld range, 1:N.
"""
function inbounds(coordinate::Real, env::Environment)::Bool
    return (1 .≤ coordinate .≤ env.gridsize)
end

"""
    transition_and_reward!(state::State, action::Integer, env::Environment; in_place = true)

Computes the state and reward for a given action. Notably this function simply points to the environment-specific implementations (open world or waterworld).
"""
function transition_and_reward!(state::State, action::Integer, env::Environment; in_place = true)
    if env.environment_type == :open
        return transition_and_reward_open!(state, action, env; in_place = in_place)
    elseif env.environment_type == :waterworld
        return transition_and_reward_ww!(state, action, env; in_place = in_place)
    end
end


"""
    map_after_move!(state::State, action::Integer, env::Environment; in_place = true)

The function that updates the map in the various implementations used in "transition_and_reward!".
"""
function map_after_move!(state::State, action::Integer, env::Environment; in_place = true)
    new_tile_counter = 0
    #This may look inefficient but the compiler should take care of it.
    position = state.agent_position
    map = state.map

    if !in_place
        new_map = copy.(map)
    else
        new_map = map
    end

    #The code below is ugly, but all it does is check the tiles that we moved closer to (and now can "see") to see if we've seen them before (or if they're out of bounds)
    if action == 1
        for i ∈ (-env.mapping_radius):(env.mapping_radius)
            row_nr = position[1] - (1 + env.mapping_radius-abs(i))
            col_nr = position[2] + i
            if inbounds(row_nr, env)&&inbounds(col_nr, env)&&iszero(map[row_nr, col_nr])
                new_tile_counter += 1
                new_map[row_nr, col_nr] = env.underlying_map[row_nr, col_nr]
            end
        end
    elseif action == 2
        for i ∈ (-env.mapping_radius):(env.mapping_radius)
            row_nr = position[1] + i
            col_nr = position[2] + (1 + env.mapping_radius-abs(i))
            if inbounds(row_nr, env)&&inbounds(col_nr, env)&&iszero(map[row_nr, col_nr])
                new_tile_counter += 1
                new_map[row_nr, col_nr] = env.underlying_map[row_nr, col_nr]
            end
        end
    elseif action == 3
        for i ∈ (-env.mapping_radius):(env.mapping_radius)
            row_nr = position[1] + (1 + env.mapping_radius-abs(i))
            col_nr = position[2] + i
            if inbounds(row_nr, env)&&inbounds(col_nr, env)&&iszero(map[row_nr, col_nr])
                new_tile_counter += 1
                new_map[row_nr, col_nr] = env.underlying_map[row_nr, col_nr]
            end
        end
    elseif action == 4
        for i ∈ (-env.mapping_radius):(env.mapping_radius)
            row_nr = position[1] + i
            col_nr = position[2] - (1 + env.mapping_radius-abs(i))
            if inbounds(row_nr, env)&&inbounds(col_nr, env)&&iszero(map[row_nr, col_nr])
                new_tile_counter += 1
                new_map[row_nr, col_nr] = env.underlying_map[row_nr, col_nr]
            end
        end
    else
        error("Improper action!")
    end
    if in_place
        return new_tile_counter
    else
        return new_map, new_tile_counter
    end    
end