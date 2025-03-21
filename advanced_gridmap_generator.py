import numpy as np
import random
from queue import Queue
import matplotlib.pyplot as plt

def generate_rooms_gridmap(N, seed=None, num_rooms=None):
    """
    Generate a grid map with connected rooms.
    
    Parameters:
    - N: Size of the grid (N x N)
    - seed: Random seed for reproducibility
    - num_rooms: Number of rooms to generate (default: auto-calculated based on grid size)
    
    Returns:
    - grid: NxN numpy array where:
        0 = explorable area
        1 = inaccessible area (wall)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize grid with all walls
    grid = np.ones((N, N), dtype=np.int8)
    
    # Determine number of rooms based on grid size if not specified
    if num_rooms is None:
        num_rooms = max(3, N // 5)
    
    # Generate rooms
    rooms = []
    attempts = 0
    max_attempts = 100
    
    # Ensure the middle of the grid is accessible by creating a room there
    middle_x, middle_y = N // 2, N // 2
    middle_width = min(6, N // 4)
    middle_height = min(6, N // 4)
    middle_room = (middle_x - middle_width // 2, middle_y - middle_height // 2, middle_width, middle_height)
    rooms.append(middle_room)
    
    # Create the middle room
    x, y, width, height = middle_room
    for i in range(x, x + width):
        for j in range(y, y + height):
            if 0 <= i < N and 0 <= j < N:  # Ensure we're within grid bounds
                grid[i, j] = 0
    
    while len(rooms) < num_rooms and attempts < max_attempts:
        # Random room size
        width = random.randint(3, min(8, N//3))
        height = random.randint(3, min(8, N//3))
        
        # Random position
        x = random.randint(1, N - width - 1)
        y = random.randint(1, N - height - 1)
        
        # Check if room overlaps with existing rooms
        overlap = False
        new_room = (x, y, width, height)
        
        for room in rooms:
            rx, ry, rw, rh = room
            if (x < rx + rw + 1 and x + width + 1 > rx and
                y < ry + rh + 1 and y + height + 1 > ry):
                overlap = True
                break
        
        if not overlap:
            rooms.append(new_room)
            # Create room
            for i in range(x, x + width):
                for j in range(y, y + height):
                    grid[i, j] = 0
        
        attempts += 1
    
    # Connect rooms with corridors
    for i in range(len(rooms) - 1):
        # Get center points of rooms
        x1, y1, w1, h1 = rooms[i]
        x2, y2, w2, h2 = rooms[i + 1]
        
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        
        # Create L-shaped corridor
        if random.random() < 0.5:
            # Horizontal then vertical
            for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                grid[x, cy1] = 0
            for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                grid[cx2, y] = 0
        else:
            # Vertical then horizontal
            for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                grid[cx1, y] = 0
            for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                grid[x, cy2] = 0
    
    return grid

def generate_island_gridmap(N, seed=None, num_islands=None):
    """
    Generate a grid map with multiple islands/continents separated by water.
    
    Parameters:
    - N: Size of the grid (N x N)
    - seed: Random seed for reproducibility
    - num_islands: Number of islands to generate (default: auto-calculated)
    
    Returns:
    - grid: NxN numpy array where:
        0 = explorable area (land)
        1 = inaccessible area (water)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize grid with all water
    grid = np.ones((N, N), dtype=np.int8)
    
    # Determine number of islands
    if num_islands is None:
        num_islands = random.randint(2, 5)
    
    # Ensure the middle is accessible by creating an island there first
    center_x, center_y = N // 2, N // 2
    size = random.randint(N//6, N//3)
    
    # Create initial island at the center
    for i in range(max(0, center_x - size), min(N, center_x + size)):
        for j in range(max(0, center_y - size), min(N, center_y + size)):
            if ((i - center_x)**2 + (j - center_y)**2 <= size**2 and 
                random.random() < 0.7):  # Higher probability to ensure center is filled
                grid[i, j] = 0
    
    # Ensure the exact center is accessible
    grid[center_x, center_y] = 0
    
    # Generate additional islands
    for island in range(num_islands - 1):  # -1 because we already created one island
        # Random island center (away from the center)
        while True:
            island_x = random.randint(N//6, 5*N//6)
            island_y = random.randint(N//6, 5*N//6)
            # Ensure this island is not too close to the center
            if abs(island_x - center_x) > N//6 or abs(island_y - center_y) > N//6:
                break
        
        # Random island size
        size = random.randint(N//6, N//3)
        
        # Create initial random island shape
        for i in range(max(0, island_x - size), min(N, island_x + size)):
            for j in range(max(0, island_y - size), min(N, island_y + size)):
                if ((i - island_x)**2 + (j - island_y)**2 <= size**2 and 
                    random.random() < 0.6):
                    grid[i, j] = 0
    
    # Apply cellular automata to smooth all islands
    for _ in range(2):
        new_grid = grid.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                # Count land neighbors
                neighbors = sum(grid[i+di, j+dj] == 0 
                               for di in [-1, 0, 1] 
                               for dj in [-1, 0, 1] 
                               if 0 <= i+di < N and 0 <= j+dj < N)
                
                if grid[i, j] == 0:  # Land
                    if neighbors < 4:  # Not enough land neighbors
                        new_grid[i, j] = 1  # Convert to water
                else:  # Water
                    if neighbors > 5:  # Many land neighbors
                        new_grid[i, j] = 0  # Convert to land
        grid = new_grid
    
    # Ensure the center is still accessible after smoothing
    grid[center_x, center_y] = 0
    
    # Add some bridges between islands
    islands_connected = set()
    
    # Find all land cells
    land_cells = [(i, j) for i in range(N) for j in range(N) if grid[i, j] == 0]
    
    # Label islands
    island_labels = np.zeros((N, N), dtype=int)
    current_label = 1
    
    for i, j in land_cells:
        if island_labels[i, j] == 0:  # Unlabeled land cell
            # BFS to label connected land cells
            queue = Queue()
            queue.put((i, j))
            island_labels[i, j] = current_label
            
            while not queue.empty():
                x, y = queue.get()
                
                for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < N and 0 <= ny < N and 
                        grid[nx, ny] == 0 and island_labels[nx, ny] == 0):
                        island_labels[nx, ny] = current_label
                        queue.put((nx, ny))
            
            current_label += 1
    
    # Connect islands with bridges
    for island1 in range(1, current_label):
        for island2 in range(island1 + 1, current_label):
            if (island1, island2) not in islands_connected:
                # Find closest points between islands
                island1_cells = [(i, j) for i, j in land_cells if island_labels[i, j] == island1]
                island2_cells = [(i, j) for i, j in land_cells if island_labels[i, j] == island2]
                
                min_dist = float('inf')
                closest_pair = None
                
                # Sample some cells to avoid checking all pairs
                sample1 = random.sample(island1_cells, min(20, len(island1_cells)))
                sample2 = random.sample(island2_cells, min(20, len(island2_cells)))
                
                for i1, j1 in sample1:
                    for i2, j2 in sample2:
                        dist = abs(i1 - i2) + abs(j1 - j2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = ((i1, j1), (i2, j2))
                
                if closest_pair:
                    (i1, j1), (i2, j2) = closest_pair
                    # Create bridge
                    x, y = i1, j1
                    while (x, y) != (i2, j2):
                        if x < i2:
                            x += 1
                        elif x > i2:
                            x -= 1
                        elif y < j2:
                            y += 1
                        elif y > j2:
                            y -= 1
                        grid[x, y] = 0
                    
                    islands_connected.add((island1, island2))
                    islands_connected.add((island2, island1))
    
    return grid

def generate_simple_blocks_gridmap(N, seed=None, block_size=None):
    """
    Generate a simple grid map with four blocks of inaccessible areas near each corner.
    No border edges.
    
    Parameters:
    - N: Size of the grid (N x N)
    - seed: Random seed for reproducibility
    - block_size: Size of each block (default: auto-calculated based on grid size)
    
    Returns:
    - grid: NxN numpy array where:
        0 = explorable area
        1 = inaccessible area (blocks)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize grid with all explorable areas
    grid = np.zeros((N, N), dtype=np.int8)
    
    # Determine block size if not specified
    if block_size is None:
        block_size = max(2, N // 8)
    
    # Calculate positions for the four blocks (near corners but not touching them)
    corner_offset = max(2, N // 6)
    
    # Ensure block size and corner offset are appropriate for the grid size
    block_size = min(block_size, N // 4)
    corner_offset = min(corner_offset, N // 4)
    
    # Top-left block
    for i in range(corner_offset, corner_offset + block_size):
        for j in range(corner_offset, corner_offset + block_size):
            if 0 <= i < N and 0 <= j < N:  # Ensure we're within grid bounds
                grid[i, j] = 1
    
    # Top-right block
    for i in range(corner_offset, corner_offset + block_size):
        for j in range(N - corner_offset - block_size, N - corner_offset):
            if 0 <= i < N and 0 <= j < N:  # Ensure we're within grid bounds
                grid[i, j] = 1
    
    # Bottom-left block
    for i in range(N - corner_offset - block_size, N - corner_offset):
        for j in range(corner_offset, corner_offset + block_size):
            if 0 <= i < N and 0 <= j < N:  # Ensure we're within grid bounds
                grid[i, j] = 1
    
    # Bottom-right block
    for i in range(N - corner_offset - block_size, N - corner_offset):
        for j in range(N - corner_offset - block_size, N - corner_offset):
            if 0 <= i < N and 0 <= j < N:  # Ensure we're within grid bounds
                grid[i, j] = 1
    
    # Ensure the middle is accessible
    middle_x, middle_y = N // 2, N // 2
    grid[middle_x, middle_y] = 0
    
    return grid

def has_path(grid, start, end):
    """
    Check if there is a path from start to end in the grid using BFS.
    """
    N = grid.shape[0]
    visited = np.zeros((N, N), dtype=bool)
    
    queue = Queue()
    queue.put(start)
    visited[start] = True
    
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while not queue.empty():
        x, y = queue.get()
        
        if (x, y) == end:
            return True
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < N and 0 <= ny < N and 
                grid[nx, ny] == 0 and not visited[nx, ny]):
                visited[nx, ny] = True
                queue.put((nx, ny))
    
    return False

def print_grid(grid):
    """
    Print the grid in a readable format.
    0 = explorable (shown as '.')
    1 = inaccessible (shown as '#')
    """
    symbols = {0: '.', 1: '#'}
    for row in grid:
        print(''.join(symbols[cell] for cell in row))

def visualize_grid(grid, title="Grid Map"):
    """
    Visualize the grid using matplotlib.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='binary')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def generate_text_gridmap(N=15, seed=None, text="I ❤ RL"):
    """
    Generate a grid map with text "I ❤ RL" or custom text as the only obstacles.
    No border edges. Uses a fixed compact text size.
    
    Parameters:
    - N: Size of the grid (N x N)
    - seed: Random seed for reproducibility
    - text: Text to display on the grid (default: "I ❤ RL")
    
    Returns:
    - grid: NxN numpy array where:
        0 = explorable area
        1 = inaccessible area (text)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Fixed compact patterns for letters and symbols
    patterns = {
        'I': [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1]
        ],
        '❤': [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0]
        ],
        'R': [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 1]
        ],
        'L': [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1]
        ],
        ' ': [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ]
    }
    
    # Use minimal spacing between characters
    spacing = 1
    
    # Calculate total width needed for the text
    total_width = 0
    for char in text:
        if char in patterns:
            total_width += len(patterns[char][0]) + spacing  # Add spacing between characters
    
    # Calculate total height needed
    total_height = 5  # All patterns have height 5
    
    # Ensure N is large enough to fit the text
    min_required_N = max(total_width + 2, total_height + 2)  # Add minimal padding
    if N < min_required_N:
        N = min_required_N
        print(f"Warning: Grid size increased to {N} to fit the text")
    
    # Initialize grid with explorable areas
    grid = np.zeros((N, N), dtype=np.int8)
    
    # Calculate starting position to center the text
    start_x = N // 2 - total_height // 2  # Vertical position (middle)
    start_y = N // 2 - total_width // 2  # Horizontal position (centered)
    
    # Draw each character
    current_y = start_y
    for char in text:
        if char in patterns:
            pattern = patterns[char]
            height = len(pattern)
            width = len(pattern[0])
            
            for i in range(height):
                for j in range(width):
                    if 0 <= start_x + i < N and 0 <= current_y + j < N:  # Ensure we're within grid bounds
                        if pattern[i][j] == 1:
                            grid[start_x + i, current_y + j] = 1
            
            current_y += width + spacing  # Move to the next character position with spacing
    
    # Ensure the middle is accessible
    middle_x, middle_y = N // 2, N // 2
    grid[middle_x, middle_y] = 0
    
    return grid

def generate_rooms_with_broad_corridors(N, seed=None, num_rooms=None, corridor_width=2):
    """
    Generate a grid map with connected rooms and broader corridors.
    
    Parameters:
    - N: Size of the grid (N x N)
    - seed: Random seed for reproducibility
    - num_rooms: Number of rooms to generate (default: auto-calculated based on grid size)
    - corridor_width: Width of corridors connecting rooms (default: 2)
    
    Returns:
    - grid: NxN numpy array where:
        0 = explorable area
        1 = inaccessible area (wall)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize grid with all walls
    grid = np.ones((N, N), dtype=np.int8)
    
    # Determine number of rooms based on grid size if not specified
    if num_rooms is None:
        num_rooms = max(3, N // 5)
    
    # Generate rooms
    rooms = []
    attempts = 0
    max_attempts = 100
    
    # Ensure the middle of the grid is accessible by creating a room there
    middle_x, middle_y = N // 2, N // 2
    middle_width = min(8, N // 4)
    middle_height = min(8, N // 4)
    middle_room = (middle_x - middle_width // 2, middle_y - middle_height // 2, middle_width, middle_height)
    rooms.append(middle_room)
    
    # Create the middle room
    x, y, width, height = middle_room
    for i in range(x, x + width):
        for j in range(y, y + height):
            if 0 <= i < N and 0 <= j < N:  # Ensure we're within grid bounds
                grid[i, j] = 0
    
    # Generate additional rooms
    while len(rooms) < num_rooms and attempts < max_attempts:
        # Random room size (slightly larger rooms)
        width = random.randint(5, min(10, N//3))
        height = random.randint(5, min(10, N//3))
        
        # Random position, but ensure space for corridors
        x = random.randint(2, N - width - 2)
        y = random.randint(2, N - height - 2)
        
        # Check if room overlaps with existing rooms (with padding for corridors)
        overlap = False
        new_room = (x, y, width, height)
        
        for room in rooms:
            rx, ry, rw, rh = room
            padding = corridor_width + 1  # Extra space to ensure corridors don't overlap rooms
            if (x < rx + rw + padding and x + width + padding > rx and
                y < ry + rh + padding and y + height + padding > ry):
                overlap = True
                break
        
        if not overlap:
            rooms.append(new_room)
            # Create room
            for i in range(x, x + width):
                for j in range(y, y + height):
                    grid[i, j] = 0
        
        attempts += 1
    
    # Create a graph to track room connections
    connected = set()  # Store pairs of connected rooms
    
    # Connect all rooms to form a minimum spanning tree
    # Start with a random room
    connected_rooms = {0}  # Start with first room
    
    while len(connected_rooms) < len(rooms):
        # Find the next room to connect
        min_distance = float('inf')
        best_pair = None
        
        for i in connected_rooms:
            for j in range(len(rooms)):
                if j not in connected_rooms and (i, j) not in connected and (j, i) not in connected:
                    # Get center points of rooms
                    x1, y1, w1, h1 = rooms[i]
                    x2, y2, w2, h2 = rooms[j]
                    
                    cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
                    cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
                    
                    # Manhattan distance
                    distance = abs(cx1 - cx2) + abs(cy1 - cy2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_pair = (i, j)
        
        if best_pair:
            i, j = best_pair
            connected.add((i, j))
            connected_rooms.add(j)
            
            # Get center points of rooms
            x1, y1, w1, h1 = rooms[i]
            x2, y2, w2, h2 = rooms[j]
            
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            
            # Create broad L-shaped corridor
            half_width = corridor_width // 2
            
            if random.random() < 0.5:
                # Horizontal then vertical corridor
                # Create horizontal corridor
                for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                    for offset in range(-half_width, half_width + 1):
                        cy = cy1 + offset
                        if 0 <= x < N and 0 <= cy < N:
                            grid[x, cy] = 0
                
                # Create vertical corridor
                for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                    for offset in range(-half_width, half_width + 1):
                        cx = cx2 + offset
                        if 0 <= cx < N and 0 <= y < N:
                            grid[cx, y] = 0
            else:
                # Vertical then horizontal corridor
                # Create vertical corridor
                for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                    for offset in range(-half_width, half_width + 1):
                        cx = cx1 + offset
                        if 0 <= cx < N and 0 <= y < N:
                            grid[cx, y] = 0
                
                # Create horizontal corridor
                for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                    for offset in range(-half_width, half_width + 1):
                        cy = cy2 + offset
                        if 0 <= x < N and 0 <= cy < N:
                            grid[x, cy] = 0
    
    # Add some extra connections for loops (not just a tree)
    extra_connections = min(len(rooms) // 2, 3)  # Add a few extra connections
    
    potential_connections = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if (i, j) not in connected and (j, i) not in connected:
                # Calculate distance between room centers
                x1, y1, w1, h1 = rooms[i]
                x2, y2, w2, h2 = rooms[j]
                
                cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
                cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
                
                distance = abs(cx1 - cx2) + abs(cy1 - cy2)
                potential_connections.append((i, j, distance))
    
    # Sort by distance and add shortest ones
    potential_connections.sort(key=lambda x: x[2])
    
    for i, j, _ in potential_connections[:extra_connections]:
        connected.add((i, j))
        
        # Get center points of rooms
        x1, y1, w1, h1 = rooms[i]
        x2, y2, w2, h2 = rooms[j]
        
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        
        # Create broad L-shaped corridor
        half_width = corridor_width // 2
        
        if random.random() < 0.5:
            # Horizontal then vertical corridor
            for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                for offset in range(-half_width, half_width + 1):
                    cy = cy1 + offset
                    if 0 <= x < N and 0 <= cy < N:
                        grid[x, cy] = 0
            
            for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                for offset in range(-half_width, half_width + 1):
                    cx = cx2 + offset
                    if 0 <= cx < N and 0 <= y < N:
                        grid[cx, y] = 0
        else:
            # Vertical then horizontal corridor
            for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                for offset in range(-half_width, half_width + 1):
                    cx = cx1 + offset
                    if 0 <= cx < N and 0 <= y < N:
                        grid[cx, y] = 0
            
            for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                for offset in range(-half_width, half_width + 1):
                    cy = cy2 + offset
                    if 0 <= x < N and 0 <= cy < N:
                        grid[x, cy] = 0
    
    # Add some small random features (small rooms, alcoves) to make the map more interesting
    num_features = random.randint(3, 6)
    for _ in range(num_features):
        feature_size = random.randint(2, 4)
        feature_x = random.randint(2, N - feature_size - 2)
        feature_y = random.randint(2, N - feature_size - 2)
        
        # Only add feature if it connects to an existing open space
        has_connection = False
        for i in range(feature_x - 1, feature_x + feature_size + 1):
            for j in range(feature_y - 1, feature_y + feature_size + 1):
                if (0 <= i < N and 0 <= j < N and grid[i, j] == 0 and 
                    (i == feature_x - 1 or i == feature_x + feature_size or 
                     j == feature_y - 1 or j == feature_y + feature_size)):
                    has_connection = True
                    break
            if has_connection:
                break
        
        if has_connection:
            for i in range(feature_x, feature_x + feature_size):
                for j in range(feature_y, feature_y + feature_size):
                    if 0 <= i < N and 0 <= j < N:
                        grid[i, j] = 0
    
    return grid

# Example usage
if __name__ == "__main__":
    # Set a random seed for reproducibility
    seed = 42
    
    # Generate and visualize different types of gridmaps
    
    # 1. Rooms gridmap
    print("\nGenerating rooms gridmap...")
    rooms_grid = generate_rooms_gridmap(25, seed=seed+1)
    print_grid(rooms_grid)
    print(f"Explorable percentage: {np.sum(rooms_grid == 0) / rooms_grid.size * 100:.1f}%")
    #visualize_grid(rooms_grid, "Rooms Grid Map")
    
    # 2. Island gridmap
    print("\nGenerating island gridmap...")
    island_grid = generate_island_gridmap(30, seed=seed+2)
    print_grid(island_grid)
    print(f"Explorable percentage: {np.sum(island_grid == 0) / island_grid.size * 100:.1f}%")
    #visualize_grid(island_grid, "Island Grid Map")
    
    # 3. Simple blocks gridmap
    print("\nGenerating simple blocks gridmap...")
    blocks_grid = generate_simple_blocks_gridmap(25, seed=seed+3)
    print_grid(blocks_grid)
    print(f"Explorable percentage: {np.sum(blocks_grid == 0) / blocks_grid.size * 100:.1f}%")
    #visualize_grid(blocks_grid, "Simple Blocks Grid Map")
    
    # Test with different sizes
    print("\nGenerating small simple blocks gridmap (10x10)...")
    small_blocks_grid = generate_simple_blocks_gridmap(10, seed=seed+3)
    print_grid(small_blocks_grid)
    print(f"Explorable percentage: {np.sum(small_blocks_grid == 0) / small_blocks_grid.size * 100:.1f}%")
    
    # 4. Text gridmap "I ❤ RL"
    print("\nGenerating text gridmap...")
    text_grid = generate_text_gridmap(15, seed=seed+5)
    print_grid(text_grid)
    print(f"Explorable percentage: {np.sum(text_grid == 0) / text_grid.size * 100:.1f}%")
    print(f"Grid size: {text_grid.shape[0]}x{text_grid.shape[1]}")
    #visualize_grid(text_grid, "I ❤ RL Grid Map")

    # 5. Rooms with broad corridors
    print("\nGenerating rooms with broad corridors gridmap...")
    broad_corridors_grid = generate_rooms_with_broad_corridors(40, seed=seed+6)
    print_grid(broad_corridors_grid)
    print(f"Explorable percentage: {np.sum(broad_corridors_grid == 0) / broad_corridors_grid.size * 100:.1f}%")