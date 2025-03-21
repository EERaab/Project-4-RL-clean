#This defines the q-hat structure

@kwdef struct LinearBalancedQhat<:QhatType #unnecessary in current implementation
end


"""
    qhat(sa_features::Vector{Matrix{Float64}}, thetas::Vector{Vector{Float64}}, qhat_type::LinearBalancedQhat)

Uses state action features to compute qhat for the linear balanced model. Uses more memory than just using state_features and the matrix A(a) but does fewer computations. 
"""
function qhat(sa_features::Vector{Matrix{Float64}}, thetas::Vector{Vector{Float64}}, qhat_type::LinearBalancedQhat)
    qhat_vec = zeros(Float64, 4)
    for j ∈ eachindex(sa_features)
        for action ∈ 1:4
            qhat_vec[action] += dot(thetas[j], sa_features[j][:, action])
        end
    end
    return qhat_vec
end

mutable struct WaterworldParameters
    water_blocking_param::Float64
    beyond_params::Vector{Vector{Float64}} #bleerrhg
    unmapped_params::Vector{Vector{Float64}}  
    water_params::Vector{Vector{Float64}}  
end

"""
    qhat(sa_features::WaterworldFeatures, thetas::WaterworldParameters, qhat_type::LinearBalancedQhat)

Calculate q-hat for the given features and parameters.
"""
function qhat(sa_features::StateActionFeaturesWaterworld, thetas::WaterworldParameters, qhat_type::LinearBalancedQhat)
    qhat_vec = zeros(Float64, 4)
    qhat_vec += thetas.water_blocking_param .* sa_features.water_blocking

    for action ∈ 1:4
        for j ∈ eachindex(sa_features.beyond)            
            qhat_vec[action] += dot(thetas.beyond_params[j], sa_features.beyond[j][:, action])            
        end
        for j ∈ eachindex(sa_features.unmapped)            
            qhat_vec[action] += dot(thetas.unmapped_params[j], sa_features.unmapped[j][:, action])            
        end
        for j ∈ eachindex(sa_features.water)            
            qhat_vec[action] += dot(thetas.water_params[j], sa_features.water[j][:, action])            
        end
    end
    return qhat_vec
end


"""
    qhat_grad(sa_features::Vector{Matrix{Float64}}, thetas, action::Integer, qhat_type::LinearBalancedQhat; opt::Integer = 0)

Uses state action features in open gridworld to compute ∇qhat for the linear balanced model, but really only computes a partial gradient corresponding to the j:th-feature seet. 
Because of the substantially different structure for water-world a different function is used to update parameters there.

In non-linear models the thetas come into play, but here they do not, however the option to call the function with thetas must exist for consistent calls.
"""
function qhat_grad(sa_features::Vector{Matrix{Float64}}, theta, action::Integer, qhat_type::LinearBalancedQhat; opt::Integer = 0)
    return sa_features[opt][:, action]
end


"""
    initial_parameter_vector(param::LearningParameters)

Takes param and returns an appropriate array of parameter vectors. This will depend on the feature type and the qhat approximation. Ask Erik if you want to know whats going on here.
"""
function initial_parameter_vector_SARSA(param::LearningParameters)
    if param.environment.environment_type == :open
        if (param.qhat_type isa LinearBalancedQhat)
            if param.ignore_orthogonal_directions
                theta_length = 2
            else
                theta_length = 3
            end
        else
            error("Please select an implemented qhat approximation.")
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
            error("Please select an implemented feature type.")
        end
        return fill(zeros(Float64,theta_length),theta_num)
    elseif param.environment.environment_type == :waterworld
        #water_blocking::Vector{Float64}
        #beyond::Vector{Vector{Float64}}
        #unmapped::Vector{Vector{Float64}}
        #water::Vector{Vector{Float64}}
        if (param.qhat_type isa LinearBalancedQhat)
            if param.ignore_orthogonal_directions
                theta_length = 2
            else
                theta_length = 3
            end
        else
            error("Please select an implemented qhat approximation.")
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