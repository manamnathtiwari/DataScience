import tkinter as tk
from tkinter import ttk, messagebox  # Added messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.spatial import KDTree
import threading
import time  # Added time

# -------- Robot Class with Swarm Behaviors --------
class Robot:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position, dtype=float)
        self.target = None
        self.velocity = np.zeros(2)

    def assign_target(self, target):
        self.target = np.array(target, dtype=float)

    def move(self, neighbors, alpha=0.4, cohesion_weight=0.2, separation_weight=0.3, alignment_weight=0.1):
        if self.target is None:
            return

        to_target = self.target - self.position

        if neighbors:
            center = np.mean([n.position for n in neighbors], axis=0)
            cohesion = center - self.position
        else:
            cohesion = np.zeros(2)

        separation = np.zeros(2)
        for neighbor in neighbors:
            diff = self.position - neighbor.position
            dist = np.linalg.norm(diff)
            if 0 < dist < 5:
                separation += diff / dist

        alignment = np.mean([n.velocity for n in neighbors], axis=0) if neighbors else np.zeros(2)

        direction = (
            alpha * to_target +
            cohesion_weight * cohesion +
            separation_weight * separation +
            alignment_weight * alignment +
            np.random.normal(0, 0.01, 2)
        )

        self.velocity = direction
        self.position += 0.05 * self.velocity

# -------- Main Simulator GUI and Logic --------
class SwarmSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Swarm Robot Simulator")
        self.setup_ui()
        self.robots = []
        self.running = False

    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(control_frame, text="Number of Robots:").pack()
        self.num_robots_entry = tk.Entry(control_frame)
        self.num_robots_entry.insert(0, "1000")
        self.num_robots_entry.pack()

        tk.Label(control_frame, text="Target Shape:").pack()
        self.shape_var = tk.StringVar()
        shape_dropdown = ttk.Combobox(control_frame, textvariable=self.shape_var)
        shape_dropdown['values'] = ("Square", "Rectangle", "Circle")
        shape_dropdown.current(0)
        shape_dropdown.pack()

        tk.Label(control_frame, text="Formation Type:").pack()
        self.form_var = tk.StringVar()
        form_dropdown = ttk.Combobox(control_frame, textvariable=self.form_var)
        form_dropdown['values'] = ("Additive", "Subtractive")
        form_dropdown.current(0)
        form_dropdown.pack()

        tk.Label(control_frame, text="Interaction Range (neighbors):").pack()
        self.range_entry = tk.Entry(control_frame)
        self.range_entry.insert(0, "5")
        self.range_entry.pack()

        self.sim_button = tk.Button(control_frame, text="Start Simulation", command=self.run_simulation)
        self.sim_button.pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def generate_robots(self, count):
        return [Robot(i, np.random.rand(2) * 100) for i in range(count)]

    def generate_target_shape(self, shape, count):
        if shape == "Circle":
            angles = np.linspace(0, 2 * np.pi, count)
            radius = 30
            x = 50 + radius * np.cos(angles)
            y = 50 + radius * np.sin(angles)
        elif shape == "Square":
            side = int(np.sqrt(count))
            x = np.linspace(30, 70, side)
            y = np.linspace(30, 70, side)
            x, y = np.meshgrid(x, y)
            x, y = x.flatten(), y.flatten()
        elif shape == "Rectangle":
            width, height = int(np.sqrt(count * 2)), int(np.sqrt(count / 2))
            x = np.linspace(25, 75, width)
            y = np.linspace(35, 65, height)
            x, y = np.meshgrid(x, y)
            x, y = x.flatten(), y.flatten()
        return np.vstack((x[:count], y[:count])).T

    def assign_targets(self, robots, targets):
        positions = np.array([r.position for r in robots])
        tree = KDTree(targets)
        distances, indices = tree.query(positions)
        for i, idx in enumerate(indices):
            robots[i].assign_target(targets[idx])

    def simulate_step(self, robots, interaction_range):
        positions = np.array([r.position for r in robots])
        tree = KDTree(positions)
        for i, robot in enumerate(robots):
            _, indices = tree.query(robot.position, k=min(interaction_range + 1, len(robots)))
            neighbors = [robots[j] for j in indices if j != i]
            robot.move(neighbors)

    def run_simulation(self):
        num_robots = int(self.num_robots_entry.get())
        shape = self.shape_var.get()
        formation = self.form_var.get()
        interaction_range = int(self.range_entry.get())

        self.robots = self.generate_robots(num_robots)
        target_count = num_robots if formation == "Additive" else num_robots // 2
        targets = self.generate_target_shape(shape, target_count)

        if formation == "Subtractive":
            self.assign_targets(self.robots[:target_count], targets)
        else:
            self.assign_targets(self.robots, targets)

        self.running = True
        threading.Thread(target=self.simulate_loop, args=(interaction_range,), daemon=True).start()

    def simulate_loop(self, interaction_range):
        steps = 300
        start_time = time.time()

        for _ in range(steps):
            if not self.running:
                break
            self.simulate_step(self.robots, interaction_range)
            self.update_plot()

        self.running = False
        end_time = time.time()
        total_time = end_time - start_time

        # Display a pop-up with the total simulation time
        messagebox.showinfo("Simulation Complete", f"Simulation completed in {total_time:.2f} seconds!")

    def update_plot(self):
        self.ax.clear()
        positions = np.array([r.position for r in self.robots])
        sample_size = min(1000, len(positions))
        sample = positions[np.random.choice(len(positions), sample_size, replace=False)]
        self.ax.scatter(sample[:, 0], sample[:, 1], color='blue', s=1)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("Swarm Shape Formation")
        self.canvas.draw()

# -------- Launch the Application --------
if __name__ == "__main__":
    root = tk.Tk()
    app = SwarmSimulator(root)
    root.mainloop()
