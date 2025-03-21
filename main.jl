using Random, Plots, RollingFunctions, Statistics, LinearAlgebra, LaTeXStrings, PyCall #,BenchmarkTools

struct State
    agent_position::Vector{Int8}
    map::Matrix{Int8}
end

abstract type FeatureType
end

abstract type QhatType
end

@pyinclude("gridmap_generator.py");

include("functions/utils.jl")
include("functions/environments and learning.jl")
include("functions/basic features.jl")
include("functions/simple terrain.jl")
include("functions/plotting total mapping.jl")

#These are enough for an "open" gridworld
include("functions/open gridworld.jl")
include("functions/open world features.jl")

#These are for waterworld
include("functions/waterworld.jl")
include("functions/waterworld features.jl")

#This is for SARSA
include("functions/qhat.jl")
include("functions/SARSA.jl")

#This is for REINFORCE with soft-max policy
include("functions/soft max policy.jl")
include("functions/REINFORCE.jl")

#Running tests - VERY SLOW, dont run when starting up
#include("functions/analysing balancing tensor.jl")
#include("functions/learning parameters testing.jl")
#include("functions/feature tests.jl")

#Running scenarios
include("functions/scenarios.jl")