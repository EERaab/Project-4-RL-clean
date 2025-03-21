
### This is the open gridworld environment, where the agent can move completely freely ###

"""
    transition_and_reward_open!(state::State, action::Integer, env::Environment; in_place = true)

Takes a state and action and returns the new state and reward in an open world. If in_place is true then it will just update the state position and map, and only return the reward. This is used in REINFORCE.
"""
function transition_and_reward_open!(state::State, action::Integer, env::Environment; in_place = true)
    #This may look inefficient but the compiler should take care of it.
    pos = state.agent_position

    if !inbounds(pos, action, env)
        if in_place
            return reward_open(env, 0)
        else
            return state, reward_open(env, 0)
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
        return State(new_pos,new_map), reward_open(env, new_tile_counter)
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
        return reward_open(env, new_tile_counter)
    end
        
end

"""
    reward_open(environment::Environment, new_tile_counter::Integer)

The reward function used in "transition_and_reward_open!".
"""
function reward_open(environment::Environment, new_tile_counter::Integer)
    #Prev.: step_penalty + new_tile_counter*mapping_reward
    return new_tile_counter*environment.mapping_reward + environment.re_mapping_penalty * (2*environment.mapping_radius + 1 - new_tile_counter)
end