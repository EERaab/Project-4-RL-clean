# The features of the open world

@kwdef struct RadialFeatures<:FeatureType
    include_horizon::Bool = true
    radius_vector::Vector{Int8} = [1,2,3]
end

@kwdef struct AnnularFeatures<:FeatureType
    include_horizon::Bool = true
    inner_radius::Int8 = 1
    outer_radius::Int8 = 2
    weight_function::Function = x -> x
end


"""
    zero_in_circle(position::Vector{Int8}, map::Matrix{Int8}, d::Integer, env::Environment)

Counts the number of zero "tiles" (entries) of a "map" (matrix) in a circle around position.
"""
function zero_in_circle(position::Vector{Int8}, map::Matrix{Int8}, d::Integer, env::Environment)
    Nd = 0.0
    Ed = 0.0
    Sd = 0.0
    Wd = 0.0
    for i ∈ -d:d
        p = d - abs(i)    
        #here I'll be inefficient and allocate the vector npos = position + [i,j]. This can be improved later. Maybe the compiler deals with it
        if p ≠ 0
            for j ∈ [Int8(-p), Int8(p)]
                npos = position + [Int8(i),Int8(j)]
                if inbounds(npos, env)
                    unexplored = iszero(map[npos[1],npos[2]])
                    if unexplored
                        if i < 0 
                            Nd += 1
                            if j > 0
                                Ed += 1
                            elseif j<0
                                Wd += 1
                            end
                        elseif i > 0
                            Sd += 1
                            if j > 0
                                Ed += 1
                            elseif j<0
                                Wd += 1
                            end
                        elseif i == 0 
                            if j > 0
                                Ed += 1
                            elseif j<0
                                Wd += 1
                            end
                        end
                    end
                end
            end
        else
            npos = position + [Int8(i), Int8(0)]
            if inbounds(npos, env)
                unexplored = iszero(map[npos[1],npos[2]])
                if unexplored
                    if i < 0 
                        Nd += 1
                    elseif i > 0
                        Sd += 1
                    end
                end
            end
        end
    end
    return [Nd,Ed,Sd,Wd]
end

"""
    features_open_world(state::State, env::Environment, feature_type::RadialFeatures)

Returns the "RadialFeatures" type of features of the state in the given environment. Each feature (except for the horizon) vector counts the number of unexplored tiles in the four cardinal directions that lie on a circle.
"""
function features_open_world(state::State, env::Environment, feature_type::RadialFeatures)
    #This may look inefficient but the compiler should take care of it.
    pos = state.agent_position 
    map = state.map

    feats = Vector{Float64}[]
    if feature_type.include_horizon        
        gs_norm = (Float64(env.gridsize)^2)/2
        N = count(iszero.(map[1:(pos[1]-1),:]))/gs_norm
        E = count(iszero.(map[:, pos[2]+1:end]))/gs_norm
        S = count(iszero.(map[(pos[1]+1:end),:]))/gs_norm
        W = count(iszero.(map[:, 1:(pos[2]-1)]))/gs_norm
        push!(feats,[N,E,S,W])
    end

    d = env.mapping_radius
    for radius ∈ feature_type.radius_vector
        scanned = zero_in_circle(pos, map, d + radius, env)/(2*(d + radius - 1) + 1)
        push!(feats, scanned)
    end
    return feats
end

"""
    features_open_world(state::State, env::Environment, feature_type::AnnularFeatures)

Returns the "AnnularFeatures" type of features of the state in the given environment. Each feature vector counts the number of unexplored tiles in the four cardinal directions that lie in an annulus.
"""
function features_open_world(state::State, env::Environment, feature_type::AnnularFeatures)
    #This may look inefficient but the compiler should take care of it.
    pos = state.agent_position 
    map = state.map

    feats = Vector{Float64}[]
    if feature_type.include_horizon        
        gs_norm = (Float64(env.gridsize)^2)/2
        N = count(iszero.(map[1:(pos[1]-1),:]))/gs_norm
        E = count(iszero.(map[:, pos[2]+1:end]))/gs_norm
        S = count(iszero.(map[(pos[1]+1:end),:]))/gs_norm
        W = count(iszero.(map[:, 1:(pos[2]-1)]))/gs_norm
        push!(feats,[N,E,S,W])
    end

    d = env.gridsize
    annular_feat  = zeros(Float64, 4)
    for radius ∈ feature_type.inner_radius:feature_type.outer_radius
        annular_feat .+= (weightfunction(radius)/(2*(d + radius - 1) + 1))*zero_in_circle(pos, map, d + radius, env)
    end
    push!(feats, annular_feat)
    return feats
end

"""
    state_action_features_open_world(state::State, balancing_tensor::Array{Float64}, lps::LearningParameters)

Returns the state action features. Let the vector χ_{a,j} be the product A(a)f_j(s) where f_j(s) is the j:th set of state features. 
This function stores an array of matrices (M_1,…,M_N) where the matrix (M_j)_{ka} = (χ_{a,j})_k, i.e. the columns of M_j are the χ_{a,j}-vectors.
"""
function state_action_features_open_world(state::State, lps::LearningParameters)
    state_feats = features_open_world(state, lps.environment, lps.feature_type)
    sa_feats = Matrix{Float64}[]
    lbt = size(lps.balancing_tensor[:,:,1], 1) #ugly and bad
    for j ∈ eachindex(state_feats)
        matr = Matrix{Float64}(undef, lbt, 4) 
        for a ∈ 1:4
            matr[:,a] .= lps.balancing_tensor[:,:,a]*state_feats[j]
        end
        push!(sa_feats, matr)
    end
    return sa_feats
end