### These are just meta-functions that check and call the appropriate types in the various worlds. They're kinda dumb but there's no time to fix this (should be done through types and structs).

abstract type StateActionFeatures
end

"""
    default_features(env::Environment)

Just checks the environment and returns the appropriate default features.
"""
function default_features(env::Environment)
    if env.environment_type == :open 
        return RadialFeatures()
    elseif env.environment_type == :waterworld
        return WWRadialFeatures()
    else
        error("Unimplemented world.")
    end
end


"""
    features(state::State, lps::LearningParameters)

Returns the features according the definitions from the input learning parameters (i.e. checks the environment and feature_type).
"""
function features(state::State, lps::LearningParameters)
    if lps.environment.environment_type == :open 
        return features_open_world(state, lps.environment, lps.feature_type)
    elseif lps.environment.environment_type == :waterworld
        return features_water_world(state, lps)
    else
        error("Unimplemented world.")
    end
end


"""
    state_action_features(state::State, lps::LearningParameters)

This function returns state action features, and mostly serves as a placeholder. 

Let the vector χ_{a,j} be the product A(a)f_j(s) where f_j(s) is the j:th set of state features. 
This function stores an array of matrices (M_1,…,M_N) where the matrix (M_j)_{ka} = (χ_{a,j})_k, i.e. the columns of M_j are the χ_{a,j}-vectors.
"""
function state_action_features(state::State, lps::LearningParameters)
    # in theory we could consider very different balancing methods than setting state-action-features = A(a)f(s). 
    # This function is looking really goofy here, but if we'd want to explore different types of features this function would select the correct type, given the learning parameters.
    if lps.uses_balanced_features
        if lps.environment.environment_type == :open
            return state_action_features_open_world(state, lps) 
        elseif lps.environment.environment_type == :waterworld
            return state_action_features_water_world(state, lps)
        else
            error("Unimplemented world.")
        end
    else
        error("Parameters undefined/unimplemented!")
    end
end

