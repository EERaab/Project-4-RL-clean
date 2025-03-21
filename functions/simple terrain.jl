#Our terrain generation in Waterworld

"""
    generate_connected_matrix(;gridsize = 51, density = 0.25, threshold = 0.3, starting_rad = 3)

Generates a matrix corresponding to connected normal tiles and tries to get close to the appropriate density. Does also prejudice the generation a bit to create more dense clusters of watertiles.
"""
function generate_connected_matrix(;gridsize = 51, density = 0.25, threshold = 0.3, starting_rad = 3)
    matrix = zeros(Int8, gridsize, gridsize)
    
    directions = [[1,0],[0,1],[-1,0],[0,-1]] #ugly
    inipos = fill(Int8(ceil(gridsize/2)),2)
    stack = [inipos]
    matrix[stack[end]...] = 1
    #forbidden = [] #no longer in use, thank God, it was a bad idea
    while !isempty(stack)
        pos = pop!(stack)
        for dir in directions
            mult = 1.0
            npos = pos + dir
            #if !((nx,ny) ∈ forbidden)
                if all(1 .≤ npos .≤ gridsize) && matrix[npos...] == 0 #&& !(matrix[npos...] == -1)
                    if rand() > density*mult
                        matrix[npos...] = Int8(1)
                        push!(stack, npos)
                        mult *= 1.1 #prejudices or generation a bit
                    else
                        matrix[npos...] = Int8(2)
                        #push!(forbidden,(nx,ny)) 
                        mult *= 0.9
                    end
                end
            #end
        end
    end
    for j ∈ eachindex(matrix)
        if matrix[j] == 0
            matrix[j] = Int8(2)
        end
    end
    #the code below is too costly to apply in training. Density will fluctuate greatly for density values over 0.3-0.4
    #if abs(count(matrix .== 2)/gridsize^2 - density ) > threshold
        #return generate_connected_matrix(gridsize = gridsize, density = density, threshold = threshold)
    #end
    k = rand([1,-1])
    l = rand([1,-1])
    for i ∈ -starting_rad:starting_rad # this is dumb but easy
        rj = starting_rad - abs(i)
        for j ∈ -rj:rj
            if i == k && j == l
                matrix[inipos[1]+i, inipos[2]+j] = Int8(2)
            else
                matrix[inipos[1]+i, inipos[2]+j] = Int8(1)                
            end
        end
    end
    return matrix
end
