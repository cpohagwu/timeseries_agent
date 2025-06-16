import os
import random
import numpy as np
from typing import Dict, List, Any, Union, Tuple
import pandas as pd
from copy import deepcopy
from .tuner import ModelTuner

def calculate_diversity_score(params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
    """
    Calculate diversity score between two parameter sets.
    Higher score means more diverse.
    """
    score = 0.0
    total_params = 0
    
    for key in params1.keys():
        if key in params2:
            total_params += 1
            if isinstance(params1[key], (int, float)):
                # Normalize numerical values to 0-1 range for comparison
                param_range = max(params1[key], params2[key]) - min(params1[key], params2[key])
                if param_range > 0:
                    score += abs(params1[key] - params2[key]) / param_range
            else:
                # For non-numerical values, binary diversity score
                score += 0 if params1[key] == params2[key] else 1
                
    return score / total_params if total_params > 0 else 0

def calculate_fitness(val_avg_reward: float, val_pass_percentage: float) -> float:
    """Calculate fitness score from validation metrics."""
    return val_avg_reward * val_pass_percentage

class GeneticTuner(ModelTuner):
    """
    A tuner class that uses genetic algorithms for hyperparameter optimization.
    
    Args:
        data_df (pd.DataFrame): The time series DataFrame to use for training
        base_log_dir (str): Base directory for storing logs of different model versions
        target_column (Union[str, int]): Name or index of the target column for reward calculation
        output_size (int): Number of possible actions
        population_size (int): Size of the population in genetic algorithm
        num_generations (int): Number of generations to run
        mutation_rate (float): Probability of mutation
        elitism_count (int): Number of best individuals to preserve in each generation
        initial_temperature (float): Initial temperature for simulated annealing
        cooling_rate (float): Cooling rate for simulated annealing
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        base_log_dir: str = "logs/tuning",
        target_column: Union[str, int] = "value",
        output_size: int = 3,
        population_size: int = 20,
        num_generations: int = 10,
        mutation_rate: float = 0.1,
        elitism_count: int = 2,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
    ):
        super().__init__(data_df, base_log_dir, target_column, output_size)
        self.results_prefix = "genetic_tuning_results"
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        
        # Initialize genetic algorithm tracking
        self.best_individual = None
        self.best_score = float('-inf')
        self.current_generation = 0
        self.temperature = initial_temperature

    def generate_random_params(self, params_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Generate a random set of parameters within the given ranges."""
        params = {}
        for param_name, values in params_grid.items():
            params[param_name] = random.choice(values)
        return params

    def mutate_params(self, params: Dict[str, Any], params_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Mutate parameters with probability self.mutation_rate."""
        mutated = params.copy()
        for param_name, values in params_grid.items():
            if random.random() < self.mutation_rate:
                mutated[param_name] = random.choice(values)
        return mutated

    def crossover_params(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parameter sets."""
        child1 = {}
        child2 = {}
        
        for param_name in parent1.keys():
            if random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
                
        return child1, child2

    def select_parents(self, population: List[Dict], scores: List[float], diversity_scores: List[float], 
                      temperature: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select parents based on fitness and diversity scores."""
        # Calculate weights based on temperature
        fitness_weight = 1.0
        diversity_weight = max(0, min(1, temperature / self.initial_temperature))
        
        # Combine scores
        total_scores = [
            (fitness_weight * f + diversity_weight * d)
            for f, d in zip(scores, diversity_scores)
        ]
        
        # Normalize scores to probabilities
        total = sum(total_scores)
        if total == 0:
            probabilities = [1/len(total_scores)] * len(total_scores)
        else:
            probabilities = [s/total for s in total_scores]
        
        # Select two parents
        parents = random.choices(population, weights=probabilities, k=2)
        return parents[0], parents[1]

    def train(
        self,
        params_grid: Dict[str, List[Any]],
        base_params: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        """
        Train models using genetic algorithm optimization with diversity scoring and simulated annealing.
        
        The genetic algorithm:
        1. Maintains a population of parameter sets
        2. Evaluates each set by training and validating a model
        3. Selects parents based on both fitness and diversity scores
        4. Uses simulated annealing to gradually transition from diversity+fitness to just fitness
        5. Creates new population through crossover and mutation
        6. Preserves best individuals through elitism
        
        Args:
            params_grid: Dictionary mapping parameter names to lists of possible values
            base_params: Optional base parameters that will be used for all models
            
        Returns:
            DataFrame containing the results for each model evaluated, sorted by fitness score
        """
        if base_params is None:
            base_params = {}

        # Initialize population of random parameter sets
        population = [
            self.generate_random_params(params_grid)
            for _ in range(self.population_size)
        ]
        
        best_individual = None
        best_score = float('-inf')
        results = []
        temperature = self.initial_temperature

        # Evolution loop
        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}/{self.num_generations}")
            
            # Evaluate current population
            generation_results = []
            for model_idx, params in enumerate(population):
                print(f"\nEvaluating model {model_idx + 1}/{self.population_size}")
                print("Parameters:", params)
                
                # Evaluate the model with current parameters
                result = self.evaluate_model(
                    params=params,
                    base_params=base_params,
                    model_name=f"genetic/gen_{generation}_model_{model_idx}"
                )
                
                # Calculate fitness
                fitness = calculate_fitness(
                    result['val_avg_reward'],
                    result['val_pass_percentage']
                )
                
                # Store results
                result_entry = {
                    **result,
                    'fitness': fitness,
                    'generation': generation
                }
                generation_results.append(result_entry)
                results.append(result_entry)

                # Update best individual if necessary
                if fitness > best_score:
                    best_score = fitness
                    best_individual = params.copy()
            
            # Calculate diversity scores for the current population
            diversity_scores = []
            for i, individual in enumerate(population):
                avg_diversity = sum(
                    calculate_diversity_score(individual, other)
                    for j, other in enumerate(population)
                    if i != j
                ) / (len(population) - 1)
                diversity_scores.append(avg_diversity)
                # Append diversity score to results
                generation_results[i]['diversity_score'] = avg_diversity

            
            # Create new population
            new_population = []
            
            # Elitism: Keep best individuals
            sorted_indices = sorted(
                range(len(population)), 
                key=lambda i: generation_results[i]['fitness'],
                reverse=True
            )
            new_population.extend(population[i] for i in sorted_indices[:self.elitism_count])
            
            # Generate rest of new population
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(
                    population, 
                    [r['fitness'] for r in generation_results],
                    diversity_scores,
                    temperature
                )
                child1, child2 = self.crossover_params(parent1, parent2)
                child1 = self.mutate_params(child1, params_grid)
                child2 = self.mutate_params(child2, params_grid)
                new_population.extend([child1, child2])
            
            # Ensure population size stays constant
            new_population = new_population[:self.population_size]
            
            # Update population and temperature
            population = new_population
            temperature *= self.cooling_rate

        # Convert results to DataFrame and sort by fitness
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('fitness', ascending=False)
        
        # Store best model information
        best_result = results_df.iloc[0]
        self.best_model_checkpoint = os.path.join(best_result['model_dir'], "model.ckpt")
        self.best_model_params = {k: best_result[k] for k in params_grid.keys()}
        
        # Save results with version number
        version = self.get_next_version()
        results_path = os.path.join(self.base_log_dir, f"{self.results_prefix}_v{version}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nTuning results saved to: {results_path}")
        
        # If num_epochs_best_model is provided, continue training the best model
        num_epochs_best_model = base_params.get('num_epochs_best_model', None)
        if num_epochs_best_model is not None and num_epochs_best_model > 0:
            print(f"\nContinuing training of best model for {num_epochs_best_model} epochs...")
            best_model_params = deepcopy(base_params or {})
            best_model_params.update(self.best_model_params)
            self.continue_training_best_model(
                num_epochs=num_epochs_best_model,
                params=best_model_params,
                model_name="genetic_best_model_continued",
            )
        
        return results_df
