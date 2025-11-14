import mesa
import pandas as pd
import numpy as np
from scipy.stats import bernoulli, t
import matplotlib.pyplot as plt
import seaborn as sns
import cognitive_functions as cf
from model import rel_DMAP_model
from agents import individual
import itertools
from datetime import datetime
import os

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"batch_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("MESA 3.3.0 BATCH RUN - PARAMETER SWEEP")
print("="*60)
print(f"Output directory: {output_dir}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# Define parameter ranges
# Beta_loc: Optimism/pessimism parameter (ranges from pessimistic to optimistic)
beta_loc_values = [0.5, 1.0, 1.5, 2.0]  # Low to High optimism

# Alpha_loc: Likelihood sensitivity parameter (ranges from insensitive to sensitive)
alpha_loc_values = [0.5, 1.0, 1.5, 2.0]  # Low to High sensitivity

# Income_rank_threshold: Desperation threshold (ranges from lenient to strict)
income_rank_threshold_values = [-0.3, -0.1, 0.1, 0.3]  # Low to High threshold

# Fixed parameters (constant across all runs)
fixed_params = {
    'lambd': 0.1,
    'gamma': 0.3,
    'reward_rb': 50,
    'reward_rf': 30,
    'cost_rb': 250,
    'beta_scale': 0.5,
    'alpha_scale': 0.5,
    'min_start_wealth': 100,
    'p': 0.8,
    'width': 40,
    'height': 40,
    'num_neighbors': 20
}

# Number of steps per simulation
n_steps = 30

# Number of replications per parameter combination
n_replications = 3

print("\nParameter ranges:")
print(f"  beta_loc (optimism): {beta_loc_values}")
print(f"  alpha_loc (sensitivity): {alpha_loc_values}")
print(f"  income_rank_threshold: {income_rank_threshold_values}")
print(f"\nFixed parameters:")
for key, value in fixed_params.items():
    print(f"  {key}: {value}")
print(f"\nSteps per simulation: {n_steps}")
print(f"Replications per condition: {n_replications}")

# Calculate total number of simulations
total_conditions = len(beta_loc_values) * len(alpha_loc_values) * len(income_rank_threshold_values)
total_simulations = total_conditions * n_replications
print(f"\nTotal parameter combinations: {total_conditions}")
print(f"Total simulations: {total_simulations}")
print("="*60)

# Create all parameter combinations
param_combinations = list(itertools.product(
    beta_loc_values,
    alpha_loc_values,
    income_rank_threshold_values
))

# Initialize lists to store results
all_model_data = []
all_agent_data = []
all_table_data = []

# Run simulations
simulation_count = 0
for beta_loc, alpha_loc, income_rank_threshold in param_combinations:
    for replication in range(n_replications):
        simulation_count += 1
        
        print(f"\n[{simulation_count}/{total_simulations}] Running simulation:")
        print(f"  beta_loc={beta_loc}, alpha_loc={alpha_loc}, threshold={income_rank_threshold}, rep={replication+1}")
        
        # Create model with current parameters
        model = rel_DMAP_model(
            beta_loc=beta_loc,
            alpha_loc=alpha_loc,
            income_rank_threshold=income_rank_threshold,
            **fixed_params
        )
        
        # Run simulation
        for step in range(n_steps):
            model.step()
            if (step + 1) % 10 == 0:
                print(f"    Step {step + 1}/{n_steps}")
        
        # Collect data
        model_df = model.datacollector.get_model_vars_dataframe()
        agent_df = model.datacollector.get_agent_vars_dataframe()
        table_df = model.datacollector.get_table_dataframe("table")
        
        # Add parameter information to dataframes
        model_df['beta_loc'] = beta_loc
        model_df['alpha_loc'] = alpha_loc
        model_df['income_rank_threshold'] = income_rank_threshold
        model_df['replication'] = replication
        model_df['simulation_id'] = simulation_count
        
        agent_df['beta_loc'] = beta_loc
        agent_df['alpha_loc'] = alpha_loc
        agent_df['income_rank_threshold'] = income_rank_threshold
        agent_df['replication'] = replication
        agent_df['simulation_id'] = simulation_count
        
        table_df = table_df.reset_index()
        table_df['beta_loc'] = beta_loc
        table_df['alpha_loc'] = alpha_loc
        table_df['income_rank_threshold'] = income_rank_threshold
        table_df['replication'] = replication
        table_df['simulation_id'] = simulation_count
        
        # Append to lists
        all_model_data.append(model_df)
        all_agent_data.append(agent_df)
        all_table_data.append(table_df)
        
        # Print summary statistics for this run
        final_crime = model_df['Proportion crime'].iloc[-1]
        final_gini = model_df['Gini'].iloc[-1]
        print(f"    Final crime proportion: {final_crime:.3f}")
        print(f"    Final Gini coefficient: {final_gini:.3f}")

print("\n" + "="*60)
print("COMBINING RESULTS")
print("="*60)

# Combine all results
combined_model_df = pd.concat(all_model_data, ignore_index=True)
combined_agent_df = pd.concat(all_agent_data, ignore_index=True)
combined_table_df = pd.concat(all_table_data, ignore_index=True)

print(f"Combined model data shape: {combined_model_df.shape}")
print(f"Combined agent data shape: {combined_agent_df.shape}")
print(f"Combined table data shape: {combined_table_df.shape}")

# Save combined results
print("\nSaving results...")
combined_model_df.to_csv(f"{output_dir}/batch_model_data.csv", index=True)
combined_agent_df.to_csv(f"{output_dir}/batch_agent_data.csv", index=True)
combined_table_df.to_csv(f"{output_dir}/batch_table_data.csv", index=False)

print(f"  Saved: {output_dir}/batch_model_data.csv")
print(f"  Saved: {output_dir}/batch_agent_data.csv")
print(f"  Saved: {output_dir}/batch_table_data.csv")

# Create summary statistics
print("\n" + "="*60)
print("GENERATING SUMMARY STATISTICS")
print("="*60)

# Calculate summary statistics by parameter combination
summary_stats = combined_model_df.groupby(['beta_loc', 'alpha_loc', 'income_rank_threshold']).agg({
    'Proportion crime': ['mean', 'std', 'min', 'max'],
    'Gini': ['mean', 'std', 'min', 'max']
}).round(4)

summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
summary_stats = summary_stats.reset_index()

# Save summary statistics
summary_stats.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
print(f"  Saved: {output_dir}/summary_statistics.csv")

# Display summary statistics
print("\nSummary Statistics (averaged across replications):")
print(summary_stats.to_string())

# Create parameter metadata file
metadata = {
    'timestamp': timestamp,
    'n_steps': n_steps,
    'n_replications': n_replications,
    'total_simulations': total_simulations,
    'beta_loc_values': beta_loc_values,
    'alpha_loc_values': alpha_loc_values,
    'income_rank_threshold_values': income_rank_threshold_values,
    'fixed_parameters': fixed_params
}

metadata_df = pd.DataFrame([metadata])
metadata_df.to_csv(f"{output_dir}/metadata.csv", index=False)
print(f"  Saved: {output_dir}/metadata.csv")

# Generate some basic visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

try:
    # Calculate mean values for final step
    final_step_data = combined_model_df[combined_model_df.index.get_level_values(0) == n_steps - 1]
    
    # Plot 1: Crime proportion by parameters
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # By beta_loc
    grouped = final_step_data.groupby('beta_loc')['Proportion crime'].agg(['mean', 'std'])
    axes[0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=5)
    axes[0].set_xlabel('Beta Loc (Optimism)', fontsize=12)
    axes[0].set_ylabel('Crime Proportion', fontsize=12)
    axes[0].set_title('Crime Proportion by Optimism Level', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # By alpha_loc
    grouped = final_step_data.groupby('alpha_loc')['Proportion crime'].agg(['mean', 'std'])
    axes[1].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=5, color='orange')
    axes[1].set_xlabel('Alpha Loc (Sensitivity)', fontsize=12)
    axes[1].set_ylabel('Crime Proportion', fontsize=12)
    axes[1].set_title('Crime Proportion by Likelihood Sensitivity', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # By income_rank_threshold
    grouped = final_step_data.groupby('income_rank_threshold')['Proportion crime'].agg(['mean', 'std'])
    axes[2].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=5, color='green')
    axes[2].set_xlabel('Income Rank Threshold', fontsize=12)
    axes[2].set_ylabel('Crime Proportion', fontsize=12)
    axes[2].set_title('Crime Proportion by Desperation Threshold', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/crime_proportion_by_parameters.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/crime_proportion_by_parameters.png")
    plt.close()
    
    # Plot 2: Gini coefficient by parameters
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # By beta_loc
    grouped = final_step_data.groupby('beta_loc')['Gini'].agg(['mean', 'std'])
    axes[0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='s', capsize=5)
    axes[0].set_xlabel('Beta Loc (Optimism)', fontsize=12)
    axes[0].set_ylabel('Gini Coefficient', fontsize=12)
    axes[0].set_title('Wealth Inequality by Optimism Level', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # By alpha_loc
    grouped = final_step_data.groupby('alpha_loc')['Gini'].agg(['mean', 'std'])
    axes[1].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='s', capsize=5, color='orange')
    axes[1].set_xlabel('Alpha Loc (Sensitivity)', fontsize=12)
    axes[1].set_ylabel('Gini Coefficient', fontsize=12)
    axes[1].set_title('Wealth Inequality by Likelihood Sensitivity', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # By income_rank_threshold
    grouped = final_step_data.groupby('income_rank_threshold')['Gini'].agg(['mean', 'std'])
    axes[2].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='s', capsize=5, color='green')
    axes[2].set_xlabel('Income Rank Threshold', fontsize=12)
    axes[2].set_ylabel('Gini Coefficient', fontsize=12)
    axes[2].set_title('Wealth Inequality by Desperation Threshold', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gini_by_parameters.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/gini_by_parameters.png")
    plt.close()
    
    # Plot 3: Heatmap of crime proportion
    pivot_data = final_step_data.groupby(['beta_loc', 'alpha_loc'])['Proportion crime'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='alpha_loc', columns='beta_loc', values='Proportion crime')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Crime Proportion'})
    plt.xlabel('Beta Loc (Optimism)', fontsize=12)
    plt.ylabel('Alpha Loc (Sensitivity)', fontsize=12)
    plt.title('Crime Proportion Heatmap: Optimism × Sensitivity', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/crime_heatmap_beta_alpha.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/crime_heatmap_beta_alpha.png")
    plt.close()
    
    print("Visualizations complete!")
    
except Exception as e:
    print(f"Warning: Could not generate all visualizations: {e}")

# Print final summary
print("\n" + "="*60)
print("BATCH RUN COMPLETE")
print("="*60)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total simulations completed: {total_simulations}")
print(f"Output directory: {output_dir}")
print("\nOutput files:")
print(f"  - batch_model_data.csv ({combined_model_df.shape[0]} rows)")
print(f"  - batch_agent_data.csv ({combined_agent_df.shape[0]} rows)")
print(f"  - batch_table_data.csv ({combined_table_df.shape[0]} rows)")
print(f"  - summary_statistics.csv")
print(f"  - metadata.csv")
print(f"  - crime_proportion_by_parameters.png")
print(f"  - gini_by_parameters.png")
print(f"  - crime_heatmap_beta_alpha.png")
print("="*60)