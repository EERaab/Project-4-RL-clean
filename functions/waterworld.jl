### This is the gridworld environment with "lakes" in the way, which hinder agent movement ###

"""
    onland(position::Vector, action::Integer, env::Environment)::Bool

Checks whether taking the action would move the agent on
"""
function onland(position::Vector, action::Integer, env::Environment)::Bool
    #this is ugly but functional and fast!
    if action == 1
        a = -1
        b = 0
    elseif action == 2
        a = 0 
        b = 1
    elseif action == 3
        a = 1
        b = 0
    elseif action == 4
        a = 0
        b = -1
    end
    return (env.underlying_map[position[1] + a,position[2] + b] == 1)
end

"""
    transition_and_reward_ww!(state::State, action::Integer, env::Environment; in_place = true)

Takes a state and action and returns the new state and reward in water world. If in_place is true then it will just update the state position and map, and only return the reward. This is used in REINFORCE.
"""
function transition_and_reward_ww!(state::State, action::Integer, env::Environment; in_place = true)
    #This may look inefficient but the compiler should take care of it.
    pos = state.agent_position

    if (!inbounds(pos, action, env))
        if in_place
            return reward_ww(env, 0)
        else
            return state, reward_ww(env, 0)
        end
    elseif !onland(pos,action, env)
        if in_place
            return reward_ww(env, 0; bumped = true)
        else
            return state, reward_ww(env, 0; bumped = true)
        end
    end
    
    if !in_place
        new_map, new_tile_counter = map_after_move!(state, action, env, in_place = false)
        if action == 1
            new_pos = [pos[1]-1, pos[2]]
        elseif action == 2
            new_pos = [pos[1], pos[2]+1]
        elseif action == 3
            new_pos = [pos[1]+1, pos[2]]
        elseif action == 4
            new_pos = [pos[1], pos[2]-1]
        else
            error("Improper action!")
        end
        return State(new_pos,new_map), reward_ww(env, new_tile_counter)
    else
        new_tile_counter = map_after_move!(state, action, env, in_place = true)
        if action == 1
            pos[1] -= 1
        elseif action == 2
            pos[2] += 1
        elseif action == 3
            pos[1] += 1
        elseif action == 4
            pos[2] -= 1
        else
            error("Improper action!")
        end
        return reward_ww(env, new_tile_counter)
    end
        
end

"""

The reward function used in "transition_and_reward_ww!".
"""
function reward_ww(environment::Environment, new_tile_counter::Integer; bumped::Bool = false)
    if bumped 
        return environment.bumping_penalty + environment.re_mapping_penalty * (2*environment.mapping_radius + 1)
    end
    return new_tile_counter*environment.mapping_reward + environment.re_mapping_penalty * (2*environment.mapping_radius + 1 - new_tile_counter)
end