#this defines the features for waterworld

@kwdef struct WWRadialFeatures<:FeatureType
    include_horizon::Bool = true
    radius_vector::Vector{Int8} = [1, 2]
end

"""
    positional_increments!(vec::Vector{Integer}, i::Integer, j::Integer)

Given a pair i,j corresponding to a position of a tile of a particular type the function increments a vector counting the number of such tiles.
"""
function positional_increments!(vec::Vector, i::Integer, j::Integer)
    if i < 0 
        vec[1] += 1
        if j > 0
            vec[2] += 1
        elseif j<0
            vec[4] += 1
        end
    elseif i > 0
        vec[3] += 1
        if j > 0
            vec[2] += 1
        elseif j<0
            vec[4] += 1
        end
    elseif i == 0 
        if j > 0
            vec[2] += 1
        elseif j<0
            vec[4] += 1
        end
    end
end

"""
    positional_increments!(vec::Vector{Integer}, i::Integer)

Given a north,south-displacement i corresponding to a position of a tile of a particular type the function increments a vector counting the number of such tiles.
"""
function positional_increments!(vec::Vector, i::Integer)
    if i < 0 
        vec[1] += 1
    elseif i > 0
        vec[3] += 1
    end
end

"""
    types_in_circle(position::Vector{Int8}, map::Matrix{Int8}, d::Integer, env::Environment)

Computes the number of tiles of the types "beyond", "unmapped" and "water" on a circle of radius d around a position.
"""
function types_in_circle(position::Vector{Int8}, map::Matrix{Int8}, d::Integer, env::Environment)
    beyond = zeros(Int,4) #N,E,S,W tiles in radius  beyond the world
    unmapped = zeros(Int,4) #N,E,S,W tiles in radius  that are unmapped
    #ground = zeros(Int,4) THIS SHOULDNT BE NECESSARY!
    water = zeros(Int,4) #N,E,S,W tiles in radius that are mapped and are known to be "water"

    #So actually there is a far more clever/general way of doing this which loops over tile-types. But whatever.
    for i ∈ -d:d
        p = d - abs(i)    
        #here I'll be inefficient and allocate the vector npos = position + [i,j]. This can be improved later. Maybe the compiler deals with it
        if p ≠ 0
            for j ∈ [Int8(-p), Int8(p)]
                npos = position + [Int8(i),Int8(j)]
                if !inbounds(npos, env) #this should have been a type-based for loop or something similar
                    #if the position is beyond the gridsize we increase the value in the beyond vector
                    positional_increments!(beyond, i, j)
                elseif map[npos[1],npos[2]] == 0 
                    #if the position is unmapped we increase the value in the unmapped vector
                    positional_increments!(unmapped, i, j)
                elseif map[npos[1],npos[2]] == 2 
                    #if the position is a water tile we increase the value in the water vector
                    positional_increments!(water, i, j)
                end
            end
        else
            npos = position + [Int8(i), Int8(0)]
            if !inbounds(npos, env) 
                #if the position is beyond the gridsize we increase the value in the beyond vector
                positional_increments!(beyond, i)
            elseif map[npos[1],npos[2]] == 0 
                #if the position is unmapped we increase the value in the unmapped vector
                positional_increments!(unmapped, i)
            elseif map[npos[1],npos[2]] == 2 
                #if the position is a water tile we increase the value in the water vector
                positional_increments!(water, i)
            end
        end
    end
    return [beyond, unmapped, water]
end

"""
    water_in_the_way(pos, direction, increment, map::Matrix)

Detects whether there is a "water" tile in the relative direction, given a map.
"""
function water_in_the_way(pos, direction, increment, map::Matrix)
    mz = size(map,1)
    if direction == 1 #this is north/south
        if pos[1] > increment
            if map[pos[1] - increment, pos[2]] == 2
                return true
            end
        end
    elseif direction == 2
        if pos[2]+increment ≤ mz
            if map[pos[1], pos[2] + increment] == 2
                return true
            end
        end
    elseif direction == 3
        if pos[1] + increment ≤ mz
            if map[pos[1] + increment, pos[2]] == 2
                return true
            end
        end
    elseif direction == 4
        if pos[2]>increment
            if map[pos[1], pos[2] - increment] == 2
                return true
            end
        end
    end
    return false
end


"""
    water_in_the_way(state::State, env::Environment)

Returns the inverse of the distance to the closest "water" tile in each cardinal direction within the mapping radius. If no such tile exists within the mapping radius returns 0.
"""
function water_in_the_way(state::State, env::Environment)
    water_blocking = zeros(Float64,4)
    for direction ∈ 1:4 #this corresponds to N, E, S, W 
        for distance ∈ 1:env.mapping_radius
            if water_in_the_way(state.agent_position, direction, distance, state.map)
                water_blocking[direction] = 1/distance
                break
            end
        end
    end
    return water_blocking
end

struct WaterworldFeatures
    water_blocking::Vector{Float64}
    beyond::Vector{Vector{Float64}}
    unmapped::Vector{Vector{Float64}}
    water::Vector{Vector{Float64}}
end

function features_water_world(state::State, env::Environment, feature_type::WWRadialFeatures)
    #This may look inefficient but the compiler should take care of it.
    pos = state.agent_position 
    map = state.map

    beyond = Vector{Float64}[]
    unmapped = Vector{Float64}[]
    water = Vector{Float64}[]

    if feature_type.include_horizon  #the checks here are really dumb, if necessary we can massively speed this up!!!!
        gs_norm = (Float64(env.gridsize)^2)/2

        N = count(iszero.(map[1:(pos[1]-1),:]))/gs_norm
        E = count(iszero.(map[:, pos[2]+1:end]))/gs_norm
        S = count(iszero.(map[(pos[1]+1:end),:]))/gs_norm
        W = count(iszero.(map[:, 1:(pos[2]-1)]))/gs_norm
        push!(unmapped,[N, E, S, W])
        
        waterN = count(map[1:(pos[1]-1),:] .== 2)/gs_norm
        waterE = count(map[:, pos[2]+1:end] .== 2)/gs_norm
        waterS = count(map[(pos[1]+1:end),:] .== 2)/gs_norm
        waterW = count(map[:, 1:(pos[2]-1)] .== 2)/gs_norm
        push!(water,[waterN, waterE, waterS, waterW])
    end

    d = env.mapping_radius
    for radius ∈ feature_type.radius_vector
        scanned = types_in_circle(pos, map, d + radius, env)/(2*(d + radius - 1) + 1) 
        #scanned is a collection of three vectors describing the number of tiles of a given sort within the radius
        #this is slow and dumb
        push!(beyond, scanned[1])
        push!(unmapped, scanned[2])
        push!(water, scanned[3]) 
    end
    return WaterworldFeatures(water_in_the_way(state, env), beyond, unmapped, water)
end

struct StateActionFeaturesWaterworld    
    water_blocking::Vector{Float64}
    #this so fucking stupid, memory is gonna bleeeeed. Should not use state-action features for larger feature types. Still, I don't have time to adjust methods for this.
    beyond::Vector{Matrix{Float64}} 
    unmapped::Vector{Matrix{Float64}} 
    water::Vector{Matrix{Float64}} 
end


function state_action_features_water_world(state::State, lps::LearningParameters)
    feats = features_water_world(state, lps.environment, lps.feature_type)
    new_sa = StateActionFeaturesWaterworld(feats.water_blocking, Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[])
    lbt = size(lps.balancing_tensor[:,:,1], 1) #ugly and bad
    for j ∈ eachindex(feats.beyond)
        matr = Matrix{Float64}(undef, lbt, 4) 
        for a ∈ 1:4
            matr[:,a] .= lps.balancing_tensor[:,:,a]*feats.beyond[j]
        end
        push!(new_sa.beyond, matr)
    end
    for j ∈ eachindex(feats.unmapped)
        matr = Matrix{Float64}(undef, lbt, 4) 
        for a ∈ 1:4
            matr[:,a] .= lps.balancing_tensor[:,:,a]*feats.unmapped[j]
        end
        push!(new_sa.unmapped, matr)
    end
    for j ∈ eachindex(feats.water)
        matr = Matrix{Float64}(undef, lbt, 4) 
        for a ∈ 1:4
            matr[:,a] .= lps.balancing_tensor[:,:,a]*feats.water[j]
        end
        push!(new_sa.water, matr)
    end
    return new_sa
end