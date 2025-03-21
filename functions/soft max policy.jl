"""
    soft_max_vector(thetas, feats, balancing_tensor)

Construct the (linear-exponent) soft-max policy vector P_a(s) = exp(⟨θ, B(a)f(s)⟩)/N where N is the probability normalization factor, i.e. N = ∑_a exp(⟨θ, B(a)f(s)⟩).

Only works for open-world, implementation for waterworld is different.
"""
function soft_max_vector(thetas::Vector, state_action_feats::Vector{Matrix{Float64}})
    hvec = zeros(Float64, 4) 
    for j ∈ eachindex(thetas)
        for action ∈ 1:4
            hvec[action] += dot(thetas[j], state_action_feats[j][:, action])
        end
    end    
    ehvec =  exp.(hvec)    
    return (ehvec/sum(ehvec))
end

"""
    gradient_ln_soft_max(policy_vector::Matrix{Float64}, state_action_feats; opt::Integer = 0)

Returns the gradient of a (linear-exponent) soft-max, i.e. with exponenetial weights h(a,s,θ) = ⟨θ, B(a)f(s)⟩ = ∑_j ⟨θ_j, A(a)f_j(s)⟩ = ∑_j ⟨θ_j, χ_{a,j}⟩ where χ_{a,j} = A(a)f_j(s) are the state action features.

Only works for open-world, implementation for waterworld is different.
"""
function gradient_ln_soft_max(policy_vector::Vector{Float64}, action::Int, state_action_feats; opt::Integer = 0)
    return state_action_feats[opt][:, action] - state_action_feats[opt]*policy_vector # this could be slow.
end

"""
    initial_parameter_vector_REINFORCE(param::LearningParameters)

Returns the vectors θ_j for REINFORCE.
"""
# THIS IS DUMB AS SHIT, structurally speaking.
#It isn't pretty but I have no time to clean it up. There is a large degree of inconsistency in using leaning parameters this way, which makes the code structure REALLY weird. 
function initial_parameter_vector_REINFORCE(param::LearningParameters)
    if param.environment.environment_type == :open
        if param.ignore_orthogonal_directions
            theta_length = 2
        else
            theta_length = 3
        end
        if param.feature_type isa RadialFeatures
            theta_num = length(param.feature_type.radius_vector)
            if param.feature_type.include_horizon
                theta_num += 1
            end
        elseif param.feature_type isa AnnularFeatures
            theta_num = 2
            if param.feature_type.include_horizon
                theta_num += 1
            end
        else
            error("Unimplemented features!")
        end
        return fill(zeros(Float64,theta_length),theta_num)

    elseif param.environment.environment_type == :waterworld
        if param.ignore_orthogonal_directions
            theta_length = 2
        else
            theta_length = 3
        end
        if param.feature_type isa WWRadialFeatures
            num = length(param.feature_type.radius_vector)
        else
            error("Unimplemented features!")
        end
        wwp  = WaterworldParameters(0.0,[],[],[])
        for j ∈ 1:num #stupid but "fill" is messing with me
            push!(wwp.beyond_params, zeros(Float64,theta_length))
            push!(wwp.unmapped_params, zeros(Float64,theta_length))
            push!(wwp.water_params, zeros(Float64,theta_length))
        end
        if param.feature_type.include_horizon
            push!(wwp.unmapped_params, zeros(Float64,theta_length))
            push!(wwp.water_params, zeros(Float64,theta_length))
        end
        return wwp
    end
end

"""
    initial_baseline_parameter_vector(lps::LearningParameters)

Unimplemented, needs to be implemented for baseline.
"""
function initial_baseline_parameter_vector(lps::LearningParameters)
    return Vector{Float64}[]
end


"""
    soft_max_vector(thetas::WaterworldParameters, sa_features::StateActionFeaturesWaterworld)

Returns the soft_max policy as a vector. 
"""
function soft_max_vector(thetas::WaterworldParameters, sa_features::StateActionFeaturesWaterworld)
    hvec = zeros(Float64, 4)
    hvec += thetas.water_blocking_param .* sa_features.water_blocking

    for action ∈ 1:4
        for j ∈ eachindex(sa_features.beyond)            
            hvec[action] += dot(thetas.beyond_params[j], sa_features.beyond[j][:, action])            
        end
        for j ∈ eachindex(sa_features.unmapped)            
            hvec[action] += dot(thetas.unmapped_params[j], sa_features.unmapped[j][:, action])            
        end
        for j ∈ eachindex(sa_features.water)            
            hvec[action] += dot(thetas.water_params[j], sa_features.water[j][:, action])            
        end
    end
    ehvec =  exp.(hvec)    
    return (ehvec/sum(ehvec))
end


"""
    soft_max_vector(thetas::WaterworldParameters, sa_features::StateActionFeaturesWaterworld)

Returns the soft_max policy as a vector.
"""
function soft_max_vector(thetas::WaterworldParameters, sa_features::StateActionFeaturesWaterworld)

    hvec = thetas.water_blocking_param .* sa_features.water_blocking

    for action ∈ 1:4
        for j ∈ eachindex(sa_features.beyond)            
            hvec[action] += dot(thetas.beyond_params[j], sa_features.beyond[j][:, action])            
        end
        for j ∈ eachindex(sa_features.unmapped)            
            hvec[action] += dot(thetas.unmapped_params[j], sa_features.unmapped[j][:, action])            
        end
        for j ∈ eachindex(sa_features.water)            
            hvec[action] += dot(thetas.water_params[j], sa_features.water[j][:, action])            
        end
    end
    ehvec =  exp.(hvec)    
    return (ehvec/sum(ehvec))
end


"""
    parameter_update_REINFORCE!(iter::Integer, δ::Float64, policy_vec::Vector{Float64}, sa_feats::StateActionFeaturesWaterworld, action::Integer, thetas, lps::LearningParameters)

Does the parameter update for the features in waterworld when REINFORCE is applied.
"""
function parameter_update_REINFORCE!(iter::Integer, δ::Float64, policy_vec::Vector{Float64}, sa_feats::StateActionFeaturesWaterworld, action::Integer, thetas, lps::LearningParameters)
    mult = δ*lps.learning_rate*(lps.discount_factor^iter)

    thetas.water_blocking_param += mult*sa_feats.water_blocking[action]

    #state_action_feats[opt][:, action] - state_action_feats[opt]*policy_vector 

    for j ∈ eachindex(thetas.beyond_params) #basically looping over different distances here
        thetas.beyond_params[j] .+= mult*(sa_feats.beyond[j][:,action] - sa_feats.beyond[j]*policy_vec)
    end
    for j ∈ eachindex(thetas.unmapped_params) #basically looping over different distances here
        thetas.unmapped_params[j] .+= mult*(sa_feats.unmapped[j][:,action] - sa_feats.unmapped[j]*policy_vec)
    end
    for j ∈ eachindex(thetas.water_params) #basically looping over different distances here
        thetas.water_params[j] .+= mult*(sa_feats.water[j][:,action] - sa_feats.water[j]*policy_vec)
    end
end