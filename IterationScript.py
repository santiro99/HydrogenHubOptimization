import time
import psutil
import os
# Choose which scenario to run: 'base', 'scenario1', ..., 'scenario5'
scenario_id = 'base'

# Dynamically import correct planning and operation modules
if scenario_id == 'base':
    from BasePlanning import run_planning
    from BaseOperation import run_operation
elif scenario_id == 'scenario1':
    from Scenario1_Planning import run_planning
    from Scenario1_Operation import run_operation
elif scenario_id == 'scenario2':
    from Scenario2_Planning import run_planning
    from Scenario2_Operation import run_operation
elif scenario_id == 'scenario3':
    from Scenario3_Planning import run_planning
    from Scenario3_Operation import run_operation
elif scenario_id == 'scenario4':
    from Scenario4_Planning import run_planning
    from Scenario4_Operation import run_operation
elif scenario_id == 'scenario5':
    from Scenario5_Planning import run_planning
    from Scenario5_Operation import run_operation
else:
    raise ValueError(f"Unknown scenario ID: {scenario_id}")

# --- LCOH Calculation Helper ---
def calculate_discounted_LCOH(CAPEX_dict, OPEX_dict, GRID_dict, H2_output_dict, phi=0.1, years=range(2025, 2051)):
    LCOH = {}
    for tech in CAPEX_dict:
        capex_discounted = sum(CAPEX_dict[tech].get(y, 0) / ((1 + phi) ** (y - 2025)) for y in years)
        opex_discounted = sum(OPEX_dict[tech].get(y, 0) / ((1 + phi) ** (y - 2025)) for y in years)
        grid_discounted = sum(GRID_dict.get(y, 0) / ((1 + phi) ** (y - 2025)) for y in years)
        h2_discounted = sum(H2_output_dict[tech].get(y, 0) / ((1 + phi) ** (y - 2025)) for y in years)
        LCOH[tech] = (capex_discounted + opex_discounted + grid_discounted) / h2_discounted if h2_discounted > 0 else 0
    return LCOH

def smooth_lcoh_update(prev_lcoh: dict, new_lcoh: dict, alpha: float):
    smoothed = {}
    threshold = 1
    for tech in prev_lcoh:
        if X_c.get(tech, 0) > threshold:  # Only smooth if tech is active
            smoothed[tech] = alpha * prev_lcoh[tech] + (1 - alpha) * new_lcoh[tech]
        else:
            smoothed[tech] = new_lcoh[tech]  # Use raw value if not installed
    return smoothed

def apply_lcoe_to_lcoh(LCOH_dict, tech_to_source, LCOE_dict, X_c, X_energy, threshold=1e-3, E_H2=0.05):
    adjusted = {}
    for tech, base_lcoh in LCOH_dict.items():
        source = tech_to_source.get(tech)
        if X_c.get(tech, 0) > threshold and X_energy.get(source, 0) > threshold:
            elec_cost = LCOE_dict.get(source, 0) * E_H2 / 1000  # USD/MWh → USD/kg H2
        else:
            elec_cost = 0  # No energy installed → no electricity cost applied
        adjusted[tech] = base_lcoh + elec_cost
    return adjusted


def lcos_dynamic_q(q_kg):
    """
    Computes LCOS based on total H2 stored (q_kg).
    """
    LCOS = 1000  # starting LCOS at low storage
    return max(1000, LCOS * (1000000 / (q_kg + 1000)))  # simple inverse curve


# Initial guess for LCOH in USD/kg H2
initial_LCOH = {'PEM': 4.5, 'Alkaline': 3.25, 'SMR': 2, 'CCUS90': 2}
initial_LCOE = {'PV': 44, 'Wind': 33} #USD/MWh
LCOS = 1000  # Fixed LCOS
efficiency = {'PEM': 0.8, 'Alkaline': 0.55, 'SMR': 1, 'CCUS90': 1}
discount_rate = 0.1
iteration = 0
max_iter = 120
convergence_threshold = 0.05

Xc_prev = None
start_time = time.time()
while iteration < max_iter:
    iteration += 1
    print(f"\n--- ITERATION {iteration} ---")

    prev_LCOH = initial_LCOH.copy()
    # PLANNING receives LCOE, LCOH, LCOS
    X_c, X_energy, X_storage, X_water, D_y, CAPEX_info, CAPEX_info_rnw, Xc_full, X_energy_per_year = run_planning(initial_LCOH, initial_LCOE, LCOS, Xc_prev)
    years = sorted(D_y.keys())

    # OPERATION receives installed capacities and gives OPEX results
    total_opex, opex_per_tech, h2_output_per_tech, grid_opex_per_year, grid_import_year, renewable_energy_yearly = run_operation(X_c, X_energy, X_storage, X_water, D_y, CAPEX_info)

    for tech in CAPEX_info:
        for y in years:
            annual_OM = 0.02 * CAPEX_info[tech].get(y, 0)  # 3% O&M cost
            opex_per_tech[tech][y] += annual_OM

    print(f"\nIteration {iteration}")
    print("Total CAPEX per technology:")
    for tech in CAPEX_info:
        print(f"  {tech}: {sum(CAPEX_info[tech].values()):,.2f} USD")

    print("\nCAPEX per Year by Technology:")
    for year in sorted(years):
        print(f"Year {year}:")
        for tech in CAPEX_info:
            capex = CAPEX_info[tech].get(year, 0)
            print(f"  {tech:<8}: {capex:,.2f} USD")

    print(f"\nIteration {iteration}")
    print("Total CAPEX per renewable source:")
    for tech in CAPEX_info_rnw:
        print(f"  {tech}: {sum(CAPEX_info_rnw[tech].values()):,.2f} USD")

    print(f"\nIteration {iteration}: Total discounted OPEX = {total_opex:,.2f} USD")
    print("Total OPEX per technology:")
    for tech in opex_per_tech:
        print(f"  {tech}: {sum(opex_per_tech[tech].values()):,.2f} USD")

    print("\nMax Installed Capacities by Technology:")
    for tech, yearly_data in Xc_full.items():
        max_year = max(yearly_data, key=yearly_data.get)
        max_value = yearly_data[max_year]
        if tech in ['PEM', 'Alkaline']:
            unit = "GW"
        else:
            unit = "ton H₂/hour"
        print(f"{tech}: {max_value/1000:,.2f} {unit} in {max_year}")

    print("\nGrid OPEX (Annualized):")
    for year in grid_opex_per_year:
        grid_op = grid_opex_per_year[year]
        print(f"  {year}: {grid_op:,.2f} USD")

    print("\nH₂ Output per Technology (Annualized):")
    for tech in h2_output_per_tech:
        total_h2 = sum(h2_output_per_tech[tech].values())/1000000
        print(f"  {tech}: {total_h2:,.2f} kT H₂")

    print("\nAnnual H₂ Production per Technology (kg H₂/year):")
    for year in sorted(D_y.keys()):
        print(f"Year {year}:")
        for tech in h2_output_per_tech:
            h2 = h2_output_per_tech[tech].get(year, 0)
            print(f"  {tech:<8}: {h2:,.2f} kg H₂")

    print("\nHydrogen Demand vs. Total Output:")
    for y in sorted(D_y.keys()):
        demand = D_y[y]
        total_output = sum(h2_output_per_tech.get(tech, {}).get(y, 0) for tech in h2_output_per_tech)
        print(f"Year {y}: Demand = {demand:,.2f} kg, Total Output = {total_output:,.2f} kg")

    # Step 1: Compute new raw LCOH (from CAPEX + OPEX + H2)
    updated_LCOH_raw = calculate_discounted_LCOH(
        CAPEX_dict=CAPEX_info,
        OPEX_dict=opex_per_tech,
        GRID_dict = grid_opex_per_year,
        H2_output_dict=h2_output_per_tech,
        phi=discount_rate,
        years=years
    )

    # === Capacity change check ===
    capacities_changed = (Xc_prev is None or any(abs(Xc_full[tech][y] - Xc_prev[tech].get(y, 0)) > 1e-2 for tech in Xc_full for y in Xc_full[tech]))
    # any(abs(X_c[tech] - Xc_prev.get(tech, 0)) > 1e-2 for tech in X_c))

    # Save X_c for next comparison
    Xc_prev = Xc_full.copy()

    # === LCOH Update ===
    if capacities_changed:
        smoothed_LCOH = smooth_lcoh_update(initial_LCOH, updated_LCOH_raw, alpha=0.7)
    else:
        smoothed_LCOH = updated_LCOH_raw  # If stuck, use raw to allow shift

    # Add LCOE electricity cost based on energy tech used
    tech_to_source = {'PEM': 'Wind', 'Alkaline': 'PV'}
    prev_LCOH = initial_LCOH.copy()
    LCOE_LCOH = apply_lcoe_to_lcoh(smoothed_LCOH, tech_to_source, initial_LCOE, X_c, X_energy)
    initial_LCOH = smoothed_LCOH



    print("\n Updated LCOH per Technology (USD/kg H₂):")
    for tech, lcoh in LCOE_LCOH.items():
        print(f"  {tech:<8}: {lcoh:,.4f} USD/kg H₂")

    # Check convergence
    if iteration >= 2:
        variation_per_tech = {
            tech: abs(initial_LCOH[tech] - prev_LCOH[tech]) / prev_LCOH[tech]
            for tech in initial_LCOH
            if prev_LCOH[tech] > 0
        }

        print("\nLCOH Variations per Tech:")
        for tech, var in variation_per_tech.items():
            print(f"  {tech}: {var:.2%}")

        # Stop if ALL techs change less than threshold
        if all(var < convergence_threshold for var in variation_per_tech.values()):
            end_time = time.time()
            elapsed = end_time - start_time
            process = psutil.Process(os.getpid())
            mem_used_mb = process.memory_info().rss / 1e6
            print(f"Total Runtime: {elapsed:.2f} seconds")
            print(f"Memory Usage: {mem_used_mb:.2f} MB")
            print("Convergence reached based on LCOH variation < threshold.")
            break





