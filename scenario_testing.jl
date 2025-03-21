
"""
The code below is used to test the parameters of the SARSA and REINFORCE algorithms for the different map scenarios and to plot the results.
"""


include("main.jl")
using DelimitedFiles
islandsgrid = Int8.(readdlm("maps/island_grid.csv", ','))


# Plot the distribution of parameter values
using Plots
using Statistics
using StatsPlots

# Parameters to track
params_to_track = ["water_blocking_param", "beyond_params1", "unmapped_params1", "water_params1"]
param_names = ["Water Blocking", "Beyond (dist=1)", "Unmapped (dist=1)", "Water (dist=1)"]

# Create plots
p = plot(layout=(2,2), size=(800, 600), legend=false)

for (i, param) in enumerate(params_to_track)
    values = extract_parameter(all_thetas, param)
    
    # Plot histogram for this parameter
    histogram!(p[i], values, alpha=0.7, 
              title=param_names[i], xlabel="Parameter Value", ylabel="Frequency",
              bins=10, fillcolor=:blue, linecolor=:black)
    
    # Add mean line
    vline!(p[i], [mean(values)], linewidth=2, color=:red, label="Mean")
end

# Save the plot
savefig(p, "parameter_distributions.png")
display(p)

# Also create a boxplot for each parameter
p2 = plot(size=(800, 400), legend=false, ylabel="Parameter Value")

boxplot_data = []
for param in params_to_track
    push!(boxplot_data, extract_parameter(all_thetas, param))
end

boxplot!(p2, param_names, boxplot_data, fillcolor=:lightblue, 
        linecolor=:black, whisker_width=0.5, marker=(:circle, 4, 0.3))

savefig(p2, "parameter_boxplots.png")
display(p2)
### Parametertestning och analys
include("main.jl")
using DelimitedFiles
test_grid =  Int8.(readdlm("maps/rooms_grid3.csv", ','))

pol_type = :greedy #kunde alltså ha varit :epsgreedy eller :softmax
maxl = 100  #maximal episode length




scenarios_results = Dict()
for config in configs
    for water_density in water_densities
        env = Environment(gridsize = 50, environment_type = :waterworld, water_density = water_density, mapping_reward = 1,bumping_penalty = 0, re_mapping_penalty=-1);
        lps_sarsa = LearningParameters(environment = env, step_number = 4, episode_length = 80, episode_number = 100, feature_type = config, epsilon_decaying = true)

        scenario_sarsa = run_scenario(features_results[(config, water_density)], lps_sarsa, test_grid, pol_type, max_length = maxl)
        scenarios_results[(config, water_density)] = scenario_sarsa
    end
end

# Display total cumulative reward per config and water density
p_scenario = plot(layout=(2,2), size=(800, 600), title=["Radius [1,2]" "Radius [-1,0,1]" "Radius [0,0,1]" "No Horizon, Radius [1,2]"])

for (i, config) in enumerate(configs)
    for water_density in water_densities
        total_reward = cumsum(scenarios_results[(config, water_density)][2])
        plot!(p_scenario[i], total_reward, label="Water Density: $water_density", legend=:bottomright)
    end
    xlabel!(p_scenario[i], "Step")
    ylabel!(p_scenario[i], "Total Cumulative Reward")
end

display(p_scenario)


heatmap(test_grid)
n_scenarios = 25
test_grid =  Int8.(readdlm("maps/island_grid2.csv", ','))

configs = [
    WWRadialFeatures(radius_vector = [1,2]),
    WWRadialFeatures(radius_vector = [-1,0,1]),
    WWRadialFeatures(radius_vector = [0,1]),
    WWRadialFeatures(include_horizon = false, radius_vector = [1,2])
]

water_densities = [0.1, 0.15, 0.2, 0.25, 0.3]
maxl = 100
# Create a nested dictionary to store all results
scenarios_results = Dict()
pol_type = :epsgreedy
for config in configs
    scenarios_results[config] = Dict()
    
    for water_density in water_densities
        scenarios_results[config][water_density] = []  # This will store results for all n_scenarios
        
        env = Environment(gridsize = 51, environment_type = :waterworld, water_density = water_density, 
                          mapping_reward = 1, bumping_penalty = 0, re_mapping_penalty=-1)
        lps_sarsa = LearningParameters(environment = env, step_number =4, episode_length = 80, 
                                      episode_number = 500, feature_type = config, epsilon_decaying = true)
        tr_sarsa = n_step_SARSA(param = lps_sarsa, plotting = true)
        # Run multiple scenarios with same configuration
        for i in 1:n_scenarios
            scenario_sarsa = run_scenario(tr_sarsa[1], lps_sarsa, test_grid, pol_type, max_length = maxl)
            push!(scenarios_results[config][water_density], scenario_sarsa)  # Store each scenario result
        end
    end
end

# Create plots for average performance metrics
p_avg_reward = plot(layout=(2,2), size=(800, 600), title=["Radius [1,2]" "Radius [-1,0,1]" "Radius [0,1]" "No Horizon, Radius [1,2]"])
p_std_dev = plot(layout=(2,2), size=(800, 600), title=["Standard Deviation of Cumulative Reward"])
p_completion = plot(title="Average Exploration Completion Rate", xlabel="Configuration", ylabel="% Explored", legend=:bottomright, bar_width=0.2)

# Store completion rates for bar chart
completion_rates = Dict()

for (i, config) in enumerate(configs)
    completion_rates[config] = Dict()
    
    for water_density in water_densities
        # Get all scenario results for this config and density
        all_scenarios = scenarios_results[config][water_density]
        
        # Find the maximum length across all scenarios
        max_length = maximum([length(scenario[2]) for scenario in all_scenarios])
        
        # Create arrays to store cumulative rewards at each step
        all_cumulative_rewards = zeros(max_length, n_scenarios)
        
        # Fill the arrays with data, padding shorter scenarios with last value
        for (j, scenario) in enumerate(all_scenarios)
            rewards = scenario[2]
            cumulative = cumsum(rewards)
            scenario_length = length(cumulative)
            
            all_cumulative_rewards[1:scenario_length, j] = cumulative
            
            # Pad with final value if scenario ended early
            if scenario_length < max_length
                all_cumulative_rewards[scenario_length+1:end, j] .= cumulative[end]
            end
        end
        
        # Calculate mean and standard deviation at each step
        mean_rewards = mean(all_cumulative_rewards, dims=2)[:, 1]
        std_rewards = std(all_cumulative_rewards, dims=2)[:, 1]
        
        # Plot average cumulative reward
        plot!(p_avg_reward[i], mean_rewards, ribbon=std_rewards, 
              label="Water Density: $water_density", legend=:bottomright)
        
        # Calculate exploration completion (percent of map explored)
        # Assuming the last state of each scenario has the explored map
        final_states = [scenario[1][end] for scenario in all_scenarios]
        exploration_rates = [count(s -> s != 0, state.map) / length(state.map) for state in final_states]
        completion_rates[config][water_density] = mean(exploration_rates)
    end
    
    xlabel!(p_avg_reward[i], "Step")
    ylabel!(p_avg_reward[i], "Average Cumulative Reward")
end

# Plot the completion rates
bar_positions = Dict()
for (i, config) in enumerate(configs)
    positions = [(i-1) + (j-1)*0.2 for j in 1:length(water_densities)]
    bar_positions[config] = positions
    
    bar!(p_completion, positions, 
        [completion_rates[config][wd] for wd in water_densities], 
        label=string(config), alpha=0.7)
end

xticks!(p_completion, [i-0.5+(length(water_densities)-1)*0.1 for i in 1:length(configs)], 
       ["Config $i" for i in 1:length(configs)])

# Add a legend for water densities
for (i, wd) in enumerate(water_densities)
    scatter!(p_completion, [], [], label="Water Density: $wd", color=i, marker=:circle, 
           markerstrokewidth=0, legend=:topright)
end

display(p_avg_reward)
display(p_completion)
n_scenarios = 25
test_grid = Int8.(readdlm("maps/island_grid2.csv", ','))

# Create a nested dictionary to store all results
scenarios_results = Dict()
pol_type = :softmax
for config in configs
    scenarios_results[config] = Dict()
    
    for water_density in water_densities
        scenarios_results[config][water_density] = []  # store results for all n_scenarios
        
        env = Environment(gridsize = 51, environment_type = :waterworld, water_density = water_density, 
                          mapping_reward = 1, bumping_penalty = 0, re_mapping_penalty=-1)
        lps_reinforce = LearningParameters(environment = env, episode_length = 200, 
                                           episode_number = 500, feature_type = config, epsilon_decaying = true, learning_rate = 0.1)
        tr_reinforce = REINFORCE(param = lps_reinforce, plotting = true)
        # Run multiple scenarios with same configuration
        for i in 1:n_scenarios
            scenario_reinforce = run_scenario(tr_reinforce[1], lps_reinforce, test_grid, pol_type, max_length = maxl)
            push!(scenarios_results[config][water_density], scenario_reinforce)  # Store each scenario result
        end
    end
end

# Create plots for average performance metrics
p_avg_reward = plot(layout=(2,2), size=(800, 600), title=["Radius [1,2]" "Radius [-1,0,1]" "Radius [0,1]" "No Horizon, Radius [1,2]"])
p_std_dev = plot(layout=(2,2), size=(800, 600), title=["Standard Deviation of Cumulative Reward"])
p_completion = plot(title="Average Exploration Completion Rate", xlabel="Configuration", ylabel="% Explored", legend=:bottomright, bar_width=0.2)

# Store completion rates for bar chart
completion_rates = Dict()

for (i, config) in enumerate(configs)
    completion_rates[config] = Dict()
    
    for water_density in water_densities
        # Get all scenario results for this config and density
        all_scenarios = scenarios_results[config][water_density]
        
        # Find the maximum length across all scenarios
        max_length = maximum([length(scenario[2]) for scenario in all_scenarios])
        
        # Create arrays to store cumulative rewards at each step
        all_cumulative_rewards = zeros(max_length, n_scenarios)
        
        # Fill the arrays with data, padding shorter scenarios with last value
        for (j, scenario) in enumerate(all_scenarios)
            rewards = scenario[2]
            cumulative = cumsum(rewards)
            scenario_length = length(cumulative)
            
            all_cumulative_rewards[1:scenario_length, j] = cumulative
            
            # Pad with final value if scenario ended early
            if scenario_length < max_length
                all_cumulative_rewards[scenario_length+1:end, j] .= cumulative[end]
            end
        end
        
        # Calculate mean and standard deviation at each step
        mean_rewards = mean(all_cumulative_rewards, dims=2)[:, 1]
        std_rewards = std(all_cumulative_rewards, dims=2)[:, 1]
        
        # Plot average cumulative reward
        plot!(p_avg_reward[i], mean_rewards, ribbon=std_rewards, 
              label="Water Density: $water_density", legend=:bottomright)
        
        # Calculate exploration completion (percent of map explored)
        # Assuming the last state of each scenario has the explored map
        final_states = [scenario[1][end] for scenario in all_scenarios]
        exploration_rates = [count(s -> s != 0, state.map) / length(state.map) for state in final_states]
        completion_rates[config][water_density] = mean(exploration_rates)
    end
    
    xlabel!(p_avg_reward[i], "Step")
    ylabel!(p_avg_reward[i], "Average Cumulative Reward")
end

# Plot the completion rates
bar_positions = Dict()
for (i, config) in enumerate(configs)
    positions = [(i-1) + (j-1)*0.2 for j in 1:length(water_densities)]
    bar_positions[config] = positions
    
    bar!(p_completion, positions, 
        [completion_rates[config][wd] for wd in water_densities], 
        label=string(config), alpha=0.7)
end

xticks!(p_completion, [i-0.5+(length(water_densities)-1)*0.1 for i in 1:length(configs)], 
       ["Config $i" for i in 1:length(configs)])

# Add a legend for water densities
for (i, wd) in enumerate(water_densities)
    scatter!(p_completion, [], [], label="Water Density: $wd", color=i, marker=:circle, 
           markerstrokewidth=0, legend=:topright)
end

display(p_avg_reward)
display(p_completion)
test_grid = Int8.(readdlm("maps/text_grid2.csv", ','))
# Create a circular path scenario with increased discovery radius
function create_circular_path_scenario(test_grid; discovery_radius = 4)
    # Get dimensions of the grid
    rows, cols = size(test_grid)
    
    # Start position (at the center of the grid)
    start_pos = [rows÷2, cols÷2]
    
    # Create a State object for the starting position
    initial_map = zeros(Int8, size(test_grid))
    initial_state = State(start_pos, initial_map)
    
    # Analyze the grid to find where the text (value 2) is located
    text_rows = findall(r -> any(test_grid[r, :] .== 2), 1:rows)
    text_cols = findall(c -> any(test_grid[:, c] .== 2), 1:cols)
    
    # Define the rectangle boundaries with some padding
    top = max(1, minimum(text_rows) - 3)
    bottom = min(rows, maximum(text_rows) + 3)
    left = max(1, minimum(text_cols) - 3)
    right = min(cols, maximum(text_cols) + 3)
    
    states = [initial_state]
    rewards = Float64[]
    actions = Int[]
    
    # Function to add a step in the path
    function add_step(new_pos, action)
        # Create a new map that reveals tiles around the new position
        new_map = copy(states[end].map)
        
        # INCREASED RADIUS: Mark all tiles within the discovery radius as explored
        radius = discovery_radius
        for i in -radius:radius
            for j in -radius:radius
                # Use a circular pattern instead of diamond for better visualization
                if i*i + j*j <= radius*radius  # Circular area
                    r, c = new_pos[1] + i, new_pos[2] + j
                    if 1 <= r <= rows && 1 <= c <= cols
                        new_map[r, c] = test_grid[r, c]
                    end
                end
            end
        end
        
        # Create a new state
        push!(states, State(new_pos, new_map))
        
        # Calculate reward (1 for each newly explored tile)
        newly_explored = count(i -> i != 0 && states[end-1].map[i] == 0, 1:length(new_map))
        push!(rewards, Float64(newly_explored))
        
        # Add the action
        push!(actions, action)
    end
    
    # Apply initial exploration at starting position
    add_step(start_pos, 0)  # Add initial exploration (action 0 is placeholder)
    
    # Move to the top-left corner of our rectangle
    current_pos = copy(start_pos)
    
    # Move up to the top
    while current_pos[1] > top
        current_pos[1] -= 1
        add_step(copy(current_pos), 1)  # Action 1 is up
    end
    
    # Move left to the left edge
    while current_pos[2] > left
        current_pos[2] -= 1
        add_step(copy(current_pos), 4)  # Action 4 is left
    end
    
    # Now trace the rectangle: right, down, left, up
    
    # Move right along the top
    while current_pos[2] < right
        current_pos[2] += 1
        add_step(copy(current_pos), 2)  # Action 2 is right
    end
    
    # Move down along the right
    while current_pos[1] < bottom
        current_pos[1] += 1
        add_step(copy(current_pos), 3)  # Action 3 is down
    end
    
    # Move left along the bottom
    while current_pos[2] > left
        current_pos[2] -= 1
        add_step(copy(current_pos), 4)  # Action 4 is left
    end
    
    # Move up along the left to complete the rectangle
    while current_pos[1] > top
        current_pos[1] -= 1
        add_step(copy(current_pos), 1)  # Action 1 is up
    end
    
    # Return a tuple that matches what run_scenario returns
    # Skip first state since we created it just for initial exploration
    return (states[2:end], rewards[2:end], actions[2:end])
end

# Create the circular path scenario with a larger discovery radius
scenario = create_circular_path_scenario(test_grid, discovery_radius = 5)  # Increased from 2 to 5

# Visualize using create_animation_scenario
create_animation_scenario(scenario, name="circular_path_wide_discovery")

# Print some statistics
println("Total steps: $(length(scenario[1])-1)")  # -1 because first state has no action
println("Total reward: $(sum(scenario[2]))")
println("Final exploration: $(count(x -> x != 0, scenario[1][end].map) / length(scenario[1][end].map) * 100)%")
# Function to calculate map coverage from a scenario
function calculate_map_coverage(scenario, test_grid)
    final_state = scenario[1][end]
    final_map = final_state.map
    
    # Calculate total exploration percentage
    total_cells = length(final_map)
    explored_cells = count(cell -> cell != 0, final_map)
    exploration_percentage = (explored_cells / total_cells) * 100
    
    # Calculate water vs land discovery
    water_found = count(cell -> cell == 2, final_map)
    land_found = count(cell -> cell == 1, final_map)
    
    total_water = count(cell -> cell == 2, test_grid)
    total_land = count(cell -> cell == 1, test_grid)
    
    water_percentage = (water_found / total_water) * 100
    land_percentage = (land_found / total_land) * 100
    
    return Dict(
        "total_coverage" => exploration_percentage,
        "water_coverage" => water_percentage,
        "land_coverage" => land_percentage,
        "steps" => length(scenario[1]) - 1,
        "reward" => sum(scenario[2])
    )
end

# Run multiple scenarios and store the results
function run_multiple_scenarios(config, n_runs, test_grid)
    println("Running $n_runs scenarios with $(typeof(config))...")
    
    scenarios_results = []
    coverage_metrics = []
    pol_type = :softmax
    maxl = 200
    
    for i in 1:n_runs
        println("Running scenario $i of $n_runs...")
        
        # Create environment and learning parameters
        env = Environment(gridsize = 51, environment_type = :waterworld, water_density = 0.1, 
                          mapping_reward = 1, bumping_penalty = 0, re_mapping_penalty=-1)
        
        lps_reinforce = LearningParameters(environment = env, episode_length = 200, 
                                          episode_number = 500, feature_type = config, 
                                          epsilon_decaying = true, learning_rate = 0.1)
        
        # Train agent
        tr_reinforce = REINFORCE(param = lps_reinforce, plotting = false)
        
        # Run scenario
        scenario = run_scenario(tr_reinforce[1], lps_reinforce, test_grid, pol_type, max_length = maxl)
        
        # Store results
        push!(scenarios_results, scenario)
        
        # Calculate and store coverage metrics
        metrics = calculate_map_coverage(scenario, test_grid)
        push!(coverage_metrics, metrics)
        
        # Print progress update
        println("Scenario $i completed: $(round(metrics["total_coverage"], digits=2))% coverage in $(metrics["steps"]) steps")
    end
    
    return scenarios_results, coverage_metrics
end

# Print summary statistics
function print_coverage_summary(coverage_metrics)
    total_coverages = [m["total_coverage"] for m in coverage_metrics]
    water_coverages = [m["water_coverage"] for m in coverage_metrics]
    land_coverages = [m["land_coverage"] for m in coverage_metrics]
    steps_taken = [m["steps"] for m in coverage_metrics]
    rewards = [m["reward"] for m in coverage_metrics]
    
    println("\n=== Coverage Summary ===")
    println("Total Map Coverage: $(round(mean(total_coverages), digits=2))% (±$(round(std(total_coverages), digits=2))%)")
    println("Water Coverage: $(round(mean(water_coverages), digits=2))% (±$(round(std(water_coverages), digits=2))%)")
    println("Land Coverage: $(round(mean(land_coverages), digits=2))% (±$(round(std(land_coverages), digits=2))%)")
    println("Steps Taken: $(round(mean(steps_taken), digits=1)) (±$(round(std(steps_taken), digits=1)))")
    println("Total Reward: $(round(mean(rewards), digits=1)) (±$(round(std(rewards), digits=1)))")
    
    # Find best and worst scenarios
    best_idx = argmax(total_coverages)
    worst_idx = argmin(total_coverages)
    
    println("\nBest scenario achieved $(round(total_coverages[best_idx], digits=2))% coverage")
    println("Worst scenario achieved $(round(total_coverages[worst_idx], digits=2))% coverage")
    
    return best_idx, worst_idx
end

# Plot coverage metrics
function plot_coverage_results(coverage_metrics)
    total_coverages = [m["total_coverage"] for m in coverage_metrics]
    water_coverages = [m["water_coverage"] for m in coverage_metrics]
    land_coverages = [m["land_coverage"] for m in coverage_metrics]
    steps_taken = [m["steps"] for m in coverage_metrics]
    
    # Create histogram of total coverage
    p1 = histogram(total_coverages, 
                  bins=8, 
                  title="Distribution of Map Coverage", 
                  xlabel="Coverage (%)", 
                  ylabel="Count",
                  legend=false)
    
    # Create scatter plot of coverage vs steps
    p2 = scatter(steps_taken, 
                total_coverages, 
                title="Coverage vs Steps", 
                xlabel="Steps Taken", 
                ylabel="Coverage (%)",
                legend=false)
    
    # Create bar chart of water vs land coverage
    p3 = bar(["Water", "Land"], 
            [mean(water_coverages), mean(land_coverages)],
            yerr=[std(water_coverages), std(land_coverages)],
            title="Water vs Land Coverage",
            ylabel="Coverage (%)",
            legend=false)
    
    combined_plot = plot(p1, p2, p3, layout=(1,3), size=(900, 300))
    return combined_plot
end

# Main function to run everything
function analyze_scenario_coverage(config, n_runs, test_grid)
    # Run scenarios
    scenarios_results, coverage_metrics = run_multiple_scenarios(config, n_runs, test_grid)
    
    # Print summary
    best_idx, worst_idx = print_coverage_summary(coverage_metrics)
    
    # Plot results
    plots = plot_coverage_results(coverage_metrics)
    display(plots)
    
    # Create animations for best and worst scenarios
    println("\nCreating animations for best and worst scenarios...")
    create_animation_scenario(scenarios_results[best_idx], name="best_coverage_scenario")
    create_animation_scenario(scenarios_results[worst_idx], name="worst_coverage_scenario")
    
    return scenarios_results, coverage_metrics, plots
end

# Run the analysis
# Assuming config is already defined and test_grid is loaded
n_runs = 5  # Start with a small number to test
config = WWRadialFeatures(radius_vector = [1,2])
pol_type = :softmax
maxl = 200
n = 25
scenarios_results, coverage_metrics, plots = analyze_scenario_coverage(config, n_runs, test_grid)
# Simple function to calculate map coverage from a scenario
function calculate_coverage(scenario)
    final_state = scenario[1][end]
    final_map = final_state.map
    
    # Calculate percentage of map explored
    total_cells = length(final_map)
    explored_cells = count(cell -> cell != 0, final_map)
    coverage_percentage = (explored_cells / total_cells) * 100
    
    return coverage_percentage
end

function run_reinforce_simulation(n_runs, test_grid)
    coverage_results = Float64[]
    for i in 1:n_runs
        println("Run $i/$n_runs...")
        
        env = Environment(gridsize = 51, environment_type = :waterworld, water_density = 0.2, 
                          mapping_reward = 1, bumping_penalty = 0, re_mapping_penalty=-1)
        
        lps_reinforce = LearningParameters(environment = env, episode_length = 80, 
                                          episode_number = 500, feature_type = config, 
                                          epsilon_decaying = true, learning_rate = 0.1)
        
        tr_reinforce = REINFORCE(param = lps_reinforce, plotting = false)
        
        scenario_reinforce = run_scenario(tr_reinforce[1], lps_reinforce, test_grid, :softmax, max_length = 200)
        
        # Calculate and store coverage
        coverage = calculate_coverage(scenario_reinforce)
        push!(coverage_results, coverage)
        
        println("Run $i completed - Coverage: $(round(coverage, digits=2))%")
    end
    return coverage_results
end
# Simple function to calculate map coverage from a scenario
function calculate_coverage(scenario)
    final_state = scenario[1][end]
    final_map = final_state.map
    
    # Calculate percentage of map explored
    total_cells = length(final_map)
    explored_cells = count(cell -> cell != 0, final_map)
    coverage_percentage = (explored_cells / total_cells) * 100
    
    return coverage_percentage
end

function run_sarsa_simulation(n_runs, test_grid)
    coverage_results_sarsa = Float64[]
    for i in 1:n_runs
        println("Run $i/$n_runs...")
        
    
        env = Environment(gridsize = 51, environment_type = :waterworld, water_density = 0.2, 
                          mapping_reward = 1, bumping_penalty = 0, re_mapping_penalty=-1)
        
        lps_sarsa = LearningParameters(environment = env, step_number = 4, episode_length = 80, 
                          episode_number = 500, feature_type = config, epsilon_decaying = true)

        tr_sarsa = n_step_SARSA(param = lps_sarsa, plotting = true)
        
        scenario_sarsa = run_scenario(tr_sarsa[1], lps_sarsa, test_grid, :softmax, max_length = 200)
        
        # Calculate and store coverage
        coverage = calculate_coverage(scenario_sarsa)
        push!(coverage_results_sarsa, coverage)
        
        println("Run $i completed - Coverage: $(round(coverage, digits=2))%")
    end
    return coverage_results_sarsa
end
grids = Dict(
    "island_grid2" => Int8.(readdlm("maps/island_grid2.csv", ',')),
    "blocks_grid2" => Int8.(readdlm("maps/blocks_grid2.csv", ',')),
    "broad_corridor_grid2" => Int8.(readdlm("maps/broad_corridors_grid2.csv", ',')),
    "rooms_grid2" => Int8.(readdlm("maps/rooms_grid2.csv", ',')),
    "island_grid3" => Int8.(readdlm("maps/island_grid3.csv", ','))
)

using Statistics


# Number of runs
n_runs = 10

# Dictionary to store results
results = Dict{String, Dict{String, Float64}}()

# Run simulations for each map
for (map_name, test_grid) in grids
    println("Running simulations for $map_name...")
    
    # Run SARSA simulations
    coverage_sarsa = run_sarsa_simulation(n_runs, test_grid)
    mean_coverage_sarsa = mean(coverage_sarsa)
    
    # Run REINFORCE simulations
    coverage_reinforce = run_reinforce_simulation(n_runs, test_grid)
    mean_coverage_reinforce = mean(coverage_reinforce)
    
    # Store results
    results[map_name] = Dict("SARSA" => mean_coverage_sarsa, "REINFORCE" => mean_coverage_reinforce)
    
    println("Results for $map_name - SARSA: $(round(mean_coverage_sarsa, digits=2))%, REINFORCE: $(round(mean_coverage_reinforce, digits=2))%")
end

# Display results
for (map_name, result) in results
    println("Map: $map_name")
    println("  SARSA Mean Coverage: $(round(result["SARSA"], digits=2))%")
    println("  REINFORCE Mean Coverage: $(round(result["REINFORCE"], digits=2))%")
end

