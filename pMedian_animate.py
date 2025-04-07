import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from docplex.mp.model import Model
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=0.08)

def solve_p_median(distance_matrix, demand_weights, p):
    num_facilities = len(distance_matrix)
    num_demand_points = len(distance_matrix[0])

    # Create CPLEX model
    model = Model(name="Weighted P-Median")

    # Binary decision variables
    open_var = model.binary_var_dict(range(num_facilities), name="open")
    assign_vars = model.binary_var_matrix(num_facilities, num_demand_points, name="assign")

    # Objective: Minimize weighted total distance
    model.minimize(model.sum(demand_weights[j] * distance_matrix[i][j] * assign_vars[i, j] 
                             for i in range(num_facilities) 
                             for j in range(num_demand_points)))

    # Constraint: Exactly p facilities must be opened
    model.add_constraint(model.sum(open_var[i] for i in range(num_facilities)) == p)

    # Constraint: Each demand point must be assigned to exactly one facility
    for j in range(num_demand_points):
        model.add_constraint(model.sum(assign_vars[i, j] for i in range(num_facilities)) == 1)

    # Constraint: Assignment only to open facilities
    for i in range(num_facilities):
        for j in range(num_demand_points):
            model.add_constraint(assign_vars[i, j] <= open_var[i])

    # Solve the model
    solution = model.solve(log_output=False)
    print(solution.objective_value)

    if solution:
        opened_facilities = [i for i in range(num_facilities) if open_var[i].solution_value > 0.5]
        assignments = {j: i for j in range(num_demand_points) for i in range(num_facilities) if assign_vars[i, j].solution_value > 0.5}
        return opened_facilities, assignments,solution.objective_value
    else:
        return None, None

def plot_solution_animation(facility_coords, demand_coords, demand_weights, opened_facilities, assignments,objecive):
    fig, ax = plt.subplots(figsize=(8, 6),facecolor='lightyellow')
    plt.gca().set_frame_on(False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    #ax.set_xlabel("X Coordinate")
    #ax.set_ylabel("Y Coordinate")
    #ax.set_title("Total distance = " + str(objecive), fontsize=14)
    #ax.set_title("Total distance = " + str(round(objecive,0)), fontsize=14)
    ax.set_title("Optimised total Euclidian distance = " + str(round(objecive,0))+"Units", fontsize=10,loc='left',color='green')
    ax.set_facecolor('lightyellow')
    ax.grid(True)

    # Scatter plots for facilities and demand points
    facility_plot = ax.scatter([], [], color='gray', s=150, marker='s', label="Facilities")
    opened_facility_plot = ax.scatter([], [], color='red', s=200, marker='s', label="Opened Facilities")
    demand_plot = ax.scatter(demand_coords[:, 0], demand_coords[:, 1], color='blue', s=demand_weights *100, alpha=0.7, label="Demand Points")

    # Lines for assignments
    assignment_lines = []

    def update_plot(frame):
        ax.clear()
        #ax.set_xlim(0, 100)
        #ax.set_ylim(0, 100)
        #ax.set_xlabel("X Coordinate")
        #ax.set_ylabel("Y Coordinate")
        #ax.set_title("P-Median Solution with Weighted Demand Points")
        #ax.set_title("Total Euclidian distance = " + str(round(objecive,0))+"Units", fontsize=14)
        ax.set_title("Total Euclidian distance = " + str(round(objecive,0))+"Units", fontsize=10,loc='left',color='green')
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])

        #ax.grid(True)

        # Always show demand points
        ax.scatter(demand_coords[:, 0], demand_coords[:, 1], color='lightgreen', s=demand_weights * 30, alpha=0.7, label="Demand Points")

        if frame >= 1:
            # Show all facilities
            ax.scatter(facility_coords[:, 0], facility_coords[:, 1], color='gray', s=50, marker='s', label="Facilities")
            for x, y in facility_coords:
                ab = AnnotationBbox(getImage('house-Gray.png'), (x, y), frameon=False)
                ax.add_artist(ab)

        if frame >= 2:
            # Show opened facilities in red
            open_x = [facility_coords[i, 0] for i in opened_facilities]
            open_y = [facility_coords[i, 1] for i in opened_facilities]
            ax.scatter(open_x, open_y, color='red', s=50, marker='s', label="Opened Facilities")
            for i, (x, y) in enumerate(zip(open_x, open_y)):
                ab = AnnotationBbox(getImage('house-xxl.png'), (x, y), frameon=False)
                ax.add_artist(ab)

        if frame >= 3:
            # Draw assignment lines
            for demand, facility in assignments.items():
                x1, y1 = demand_coords[demand]
                x2, y2 = facility_coords[facility]
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.7,linewidth=0.5)

        ax.legend(loc='lower right',frameon=True,framealpha=0.999).get_frame().set_facecolor('lightyellow')
        

        return []

    ani = animation.FuncAnimation(fig, update_plot, frames=4, interval=1000, repeat=True)
    plt.show()

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    num_facilities = 6
    num_demand_points = 130
    p = 3  # Number of facilities to open

    facility_coords = np.random.rand(num_facilities, 2) * 100
    demand_coords = np.random.rand(num_demand_points, 2) * 100
    demand_weights = np.random.randint(1, 15, size=num_demand_points)  # Random weights between 1 and 10

    distance_matrix = [[np.linalg.norm(facility_coords[i] - demand_coords[j]) for j in range(num_demand_points)] for i in range(num_facilities)]

    opened_facilities, assignments,objecive = solve_p_median(distance_matrix, demand_weights, p)
    print(objecive)

    if opened_facilities is not None:
        plot_solution_animation(facility_coords, demand_coords, demand_weights, opened_facilities, assignments,objecive/sum(demand_weights))
