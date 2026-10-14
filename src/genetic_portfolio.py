import copy
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# --- Chromosome Class ---
class Chromosome:
    def __init__(self, num_assets, genes=None):
        if genes is None:
            genes = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
        self.genes = genes
        self.fitness = 0.0

    def calculate_fitness(self, returns, risks, risk_aversion=1, beta=1):
        portfolio_return = np.dot(self.genes, returns)
        portfolio_risk = np.sqrt(max(0.0, np.dot(self.genes.T, np.dot(risks, self.genes))))

        # Penalty applies only when risk exceeds the threshold ("Risk Aversion" field in the UI)
        penalty = np.exp(-beta * max(0, portfolio_risk - risk_aversion))
        adjusted_return = portfolio_return * penalty

        self.fitness = adjusted_return / (portfolio_risk + 1e-6)


# --- GeneticAlgorithm Class ---
class GeneticAlgorithm:
    def __init__(self, num_assets, population_size, generations, crossover_rate, mutation_rate,
                 elitism, selection_method, tournament_size=None):
        self.num_assets = num_assets
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.current_generation = []
        self.next_generation = []
        self.best_fitness_per_gen = []
        self.avg_fitness_per_gen = []
        self._roulette_weights = None

    def initialize_population(self):
        self.current_generation = [Chromosome(self.num_assets) for _ in range(self.population_size)]

    def evaluate_fitness(self, returns, risks, risk_aversion):
        for chromosome in self.current_generation:
            chromosome.calculate_fitness(returns, risks, risk_aversion)
        # Roulette weights are shifted so negative fitness values are handled correctly
        fits = np.array([c.fitness for c in self.current_generation], dtype=float)
        shifted = fits - fits.min() + 1e-9
        self._roulette_weights = shifted / shifted.sum()

    def select_parent(self):
        if self.selection_method == "roulette_wheel":
            idx = np.random.choice(len(self.current_generation), p=self._roulette_weights)
            return self.current_generation[idx]
        elif self.selection_method == "tournament":
            tournament = random.sample(self.current_generation, self.tournament_size)
            return max(tournament, key=lambda chrom: chrom.fitness)
        raise ValueError(f"Unknown selection method: {self.selection_method}")

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate and self.num_assets > 1:
            crossover_point = random.randint(1, self.num_assets - 1)
            child_genes = np.concatenate((parent1.genes[:crossover_point], parent2.genes[crossover_point:]))
            child_genes = child_genes / np.sum(child_genes)
            return Chromosome(self.num_assets, genes=child_genes)
        # Return an independent copy so mutation never alters the parent / elites
        return copy.deepcopy(random.choice([parent1, parent2]))

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, self.num_assets - 1)
            mutation_value = np.random.uniform(-0.1, 0.1)
            genes = chromosome.genes.copy()
            genes[mutation_index] += mutation_value
            genes = np.clip(genes, 0, 1)
            total = np.sum(genes)
            if total <= 0:
                genes = np.ones(self.num_assets)
                total = np.sum(genes)
            chromosome.genes = genes / total

    def create_next_generation(self):
        self.next_generation = []
        sorted_population = sorted(self.current_generation, key=lambda chrom: chrom.fitness, reverse=True)
        # Elites are copied so later operations cannot modify them
        self.next_generation.extend(copy.deepcopy(c) for c in sorted_population[:self.elitism])

        while len(self.next_generation) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            self.next_generation.append(child)

        self.current_generation = self.next_generation

    def run(self, returns, risks, risk_aversion):
        self.best_fitness_per_gen = []
        self.avg_fitness_per_gen = []
        self.initialize_population()

        for generation in range(self.generations):
            self.evaluate_fitness(returns, risks, risk_aversion)
            fitnesses = [c.fitness for c in self.current_generation]
            self.best_fitness_per_gen.append(max(fitnesses))
            self.avg_fitness_per_gen.append(float(np.mean(fitnesses)))
            # The last generation stays evaluated, so the reported best is always valid
            if generation < self.generations - 1:
                self.create_next_generation()

        best_chromosome = max(self.current_generation, key=lambda chrom: chrom.fitness)
        portfolio_return = np.dot(best_chromosome.genes, returns)
        portfolio_risk = np.sqrt(np.dot(best_chromosome.genes.T, np.dot(risks, best_chromosome.genes)))

        return best_chromosome, portfolio_return, portfolio_risk


# --- UI Implementation ---
def show_main_page():
    intro_frame.pack_forget()
    main_frame.pack(fill=tk.BOTH, expand=True)


def run_algorithm():
    try:
        num_assets = int(num_assets_var.get())
        population_size = int(population_size_var.get())
        generations = int(generations_var.get())
        crossover_rate = float(crossover_rate_var.get())
        mutation_rate = float(mutation_rate_var.get())
        elitism = int(elitism_var.get())
        selection_method = selection_method_var.get()
        tournament_size = int(tournament_size_var.get()) if selection_method == "tournament" else None
        risk_aversion = float(risk_aversion_var.get())

        if returns_file:
            returns = np.atleast_1d(np.loadtxt(returns_file))
            num_assets = len(returns)  # file size takes priority over the typed value
        else:
            returns = np.random.uniform(0.01, 0.2, size=num_assets)

        if risks_file:
            risks = np.atleast_2d(np.loadtxt(risks_file))
            if returns_file is None:
                num_assets = risks.shape[0]
                returns = np.random.uniform(0.01, 0.2, size=num_assets)
        else:
            risks = np.random.uniform(0.01, 0.1, size=(num_assets, num_assets))
            risks = (risks + risks.T) / 2
            np.fill_diagonal(risks, 0.05)

        # Input validation
        if risks.shape != (num_assets, num_assets):
            raise ValueError(
                f"Risk matrix shape {risks.shape} does not match {num_assets} assets in the returns file."
            )
        if num_assets < 2:
            raise ValueError("At least 2 assets are required.")
        if population_size < 2 or generations < 1:
            raise ValueError("Population size must be >= 2 and generations >= 1.")
        if not (0 <= elitism <= population_size):
            raise ValueError("Elitism must be between 0 and the population size.")
        if selection_method == "tournament" and not (1 <= tournament_size <= population_size):
            raise ValueError("Tournament size must be between 1 and the population size.")

        ga = GeneticAlgorithm(
            num_assets=num_assets,
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            selection_method=selection_method,
            tournament_size=tournament_size,
        )

        best_solution, portfolio_return, portfolio_risk = ga.run(returns, risks, risk_aversion)

        # Normalize the genes to calculate the percentage allocation
        total_allocation = np.sum(best_solution.genes)
        allocation_percentages = best_solution.genes / total_allocation * 100

        # Display results in a scrollable frame
        result_frame = tk.Frame(scrollable_frame)
        result_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP)

        tk.Label(result_frame, text=f"Best Portfolio Weights: {best_solution.genes}", anchor="w").pack(fill=tk.X)
        tk.Label(result_frame, text=f"Best Fitness: {best_solution.fitness:.4f}", anchor="w").pack(fill=tk.X)
        tk.Label(result_frame, text=f"Portfolio Return: {portfolio_return:.4f}", anchor="w").pack(fill=tk.X)
        tk.Label(result_frame, text=f"Portfolio Risk: {portfolio_risk:.4f}", anchor="w").pack(fill=tk.X)

        allocation_text = ""
        for idx, weight in enumerate(allocation_percentages):
            allocation_text += f"{weight:.2f}% in Asset {idx + 1}\n"

        tk.Label(result_frame, text=f"Percentage Allocation:\n{allocation_text.strip()}", anchor="w").pack(fill=tk.X)

        figure = plt.Figure(figsize=(6, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(range(1, generations + 1), ga.best_fitness_per_gen, label="Best Fitness")
        ax.plot(range(1, generations + 1), ga.avg_fitness_per_gen, label="Average Fitness")
        ax.set_title("Fitness Evolution over Generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend()

        canvas = FigureCanvasTkAgg(figure, master=result_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        scrollable_canvas.update_idletasks()
        scrollable_canvas.config(scrollregion=scrollable_canvas.bbox("all"))
    except Exception as e:
        messagebox.showerror("Error", str(e))


# File selection functions
def select_returns_file():
    global returns_file
    returns_file = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")]) or None
    if returns_file:
        returns_file_label.config(text=f"Returns File: {returns_file.split('/')[-1]}")


def select_risks_file():
    global risks_file
    risks_file = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")]) or None
    if risks_file:
        risks_file_label.config(text=f"Risks File: {risks_file.split('/')[-1]}")


returns_file = None
risks_file = None


def toggle_tournament_size(*args):
    if selection_method_var.get() == "tournament":
        tournament_size_entry["state"] = "normal"
    else:
        tournament_size_entry["state"] = "disabled"


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Genetic Algorithm Portfolio Optimization")

    # --- Introductory Frame ---
    intro_frame = tk.Frame(root)
    intro_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(intro_frame, text="Welcome to the Genetic Algorithm Solver!", font=("Helvetica", 16)).pack(pady=20)

    info_text = (
        "Problem Definition:\n"
        "  - Portfolio Optimization: Finding the optimal weights for assets in a portfolio to maximize return and minimize risk.\n\n"
        "Encoding (Genotype):\n"
        "  - Real-Valued Vector: Each gene represents the weight of a specific asset in the portfolio.\n\n"
        "Decoding (Phenotype):\n"
        "  - Portfolio Weights: The normalized vector of weights applied to the portfolio's assets."
    )
    tk.Label(intro_frame, text=info_text, justify="left", font=("Helvetica", 12)).pack(pady=10, padx=20)

    start_button = tk.Button(intro_frame, text="Proceed to Genetic Algorithm", command=show_main_page)
    start_button.pack(pady=20)

    # --- Main Frame ---
    main_frame = tk.Frame(root)

    input_frame = tk.Frame(main_frame)
    input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    num_assets_var = tk.StringVar(value="10")
    population_size_var = tk.StringVar(value="200")
    generations_var = tk.StringVar(value="100")
    crossover_rate_var = tk.StringVar(value="0.85")
    mutation_rate_var = tk.StringVar(value="0.05")
    elitism_var = tk.StringVar(value="3")
    selection_method_var = tk.StringVar(value="roulette_wheel")
    tournament_size_var = tk.StringVar(value="3")
    risk_aversion_var = tk.StringVar(value="1")

    fields = [
        ("Number of Assets", num_assets_var),
        ("Population Size", population_size_var),
        ("Generations", generations_var),
        ("Crossover Rate", crossover_rate_var),
        ("Mutation Rate", mutation_rate_var),
        ("Elitism", elitism_var),
        ("Risk Aversion", risk_aversion_var),
    ]
    for idx, (label_text, variable) in enumerate(fields):
        tk.Label(input_frame, text=label_text).grid(row=idx, column=0, sticky="w")
        tk.Entry(input_frame, textvariable=variable).grid(row=idx, column=1)

    tk.Label(input_frame, text="Selection Method").grid(row=len(fields), column=0, sticky="w")
    selection_combobox = ttk.Combobox(input_frame, textvariable=selection_method_var, state="readonly")
    selection_combobox["values"] = ("roulette_wheel", "tournament")
    selection_combobox.grid(row=len(fields), column=1)

    tk.Label(input_frame, text="Tournament Size").grid(row=len(fields) + 1, column=0, sticky="w")
    tournament_size_entry = tk.Entry(input_frame, textvariable=tournament_size_var, state="disabled")
    tournament_size_entry.grid(row=len(fields) + 1, column=1)

    selection_method_var.trace_add("write", toggle_tournament_size)

    # File selection buttons
    tk.Button(input_frame, text="Select Returns File", command=select_returns_file).grid(row=len(fields) + 2, column=0, pady=5)
    returns_file_label = tk.Label(input_frame, text="Returns File: Not Selected")
    returns_file_label.grid(row=len(fields) + 2, column=1)

    tk.Button(input_frame, text="Select Risks File", command=select_risks_file).grid(row=len(fields) + 3, column=0, pady=5)
    risks_file_label = tk.Label(input_frame, text="Risks File: Not Selected")
    risks_file_label.grid(row=len(fields) + 3, column=1)

    run_button = tk.Button(main_frame, text="Run", command=run_algorithm)
    run_button.pack(pady=10)

    # Scrollable Frame for Results
    scrollable_canvas = tk.Canvas(main_frame)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=scrollable_canvas.yview)
    scrollable_hscrollbar = tk.Scrollbar(main_frame, orient="horizontal", command=scrollable_canvas.xview)
    scrollable_frame = tk.Frame(scrollable_canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: scrollable_canvas.configure(scrollregion=scrollable_canvas.bbox("all"))
    )

    scrollable_window = scrollable_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    scrollable_canvas.configure(yscrollcommand=scrollbar.set, xscrollcommand=scrollable_hscrollbar.set)
    scrollable_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    scrollable_hscrollbar.pack(side="bottom", fill="x")

    root.mainloop()
