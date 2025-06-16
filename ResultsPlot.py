import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

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
def calculate_discounted_LCOH(CAPEX_dict, OPEX_dict, GRID_dict, H2_output_dict, phi=0.07, years=range(2025, 2051)):
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

def apply_lcoe_to_lcoh(LCOH_dict, tech_to_source, LCOE_dict, X_c, X_energy, threshold=1e-3, E_H2=0.045):
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
    base = 1000  # starting LCOS at low storage
    calculation = base * (1000000 / (q_kg + 1000))
    LCOS_new = calculation
    return LCOS_new  # simple inverse curve

# Initial guess for LCOH in USD/kg H2
initial_LCOH = {'PEM': 4.3, 'Alkaline': 3.25, 'SMR': 2, 'CCUS90': 2}
initial_LCOE = {'PV': 44, 'Wind': 33} #USD/MWh
LCOS = 1000  # Fixed LCOS
efficiency = {'PEM': 0.83, 'Alkaline': 0.55, 'SMR': 1, 'CCUS90': 1}
discount_rate = 0.08
iteration = 0
max_iter = 5
convergence_threshold = 0.05

Xc_prev = None

while iteration < max_iter:
    iteration += 1
    print(f"\n--- ITERATION {iteration} ---")

    prev_LCOH = initial_LCOH.copy()
    # PLANNING receives LCOE, LCOH, LCOS
    #X_c, X_energy, X_storage, Xstorage_test, X_water, D_y, CAPEX_info, CAPEX_info_rnw, Xc_full = run_planning(initial_LCOH, initial_LCOE, LCOS, Xc_prev)
    X_c, X_energy, X_storage, X_water, D_y, CAPEX_info, CAPEX_info_rnw, Xc_full, X_energy_per_year = run_planning(initial_LCOH, initial_LCOE, LCOS, Xc_prev)
    years = sorted(D_y.keys())

    # OPERATION receives installed capacities and gives OPEX results
    #total_opex, opex_per_tech, h2_output_per_tech, grid_opex_per_year, grid_import_year, renewable_energy_yearly, storage_utilization_dict = run_operation(X_c, X_energy, X_storage, Xstorage_test, X_water, D_y, CAPEX_info)
    total_opex, opex_per_tech, h2_output_per_tech, grid_opex_per_year, grid_import_year, renewable_energy_yearly = run_operation(X_c, X_energy, X_storage, X_water, D_y, CAPEX_info)

    for tech in CAPEX_info:
        for y in years:
            annual_OM = 0.015 * CAPEX_info[tech].get(y, 0)  # 3% O&M cost
            opex_per_tech[tech][y] += annual_OM

    print(f"\nIteration {iteration}")
    print("Total CAPEX per technology:")
    for tech in CAPEX_info:
        print(f"  {tech}: {sum(CAPEX_info[tech].values()):,.2f} USD")

    '''print("\nCAPEX per Year by Technology:")
    for year in sorted(years):
        print(f"Year {year}:")
        for tech in CAPEX_info:
            capex = CAPEX_info[tech].get(year, 0)
            print(f"  {tech:<8}: {capex:,.2f} USD")'''

    print(f"\nIteration {iteration}")
    print("Total CAPEX per renewable source:")
    for tech in CAPEX_info_rnw:
        print(f"  {tech}: {sum(CAPEX_info_rnw[tech].values()):,.2f} USD")

    print(f"\nIteration {iteration}: Total discounted OPEX = {total_opex:,.2f} USD")
    print("Total OPEX per technology:")
    for tech in opex_per_tech:
        print(f"  {tech}: {sum(opex_per_tech[tech].values()):,.2f} USD")

    '''print("\nOPEX per Year by Technology:")
    for year in sorted(years):
        print(f"Year {year}:")
        for tech in opex_per_tech:
            opex = opex_per_tech[tech].get(year, 0)
            print(f"  {tech:<8}: {opex:,.2f} USD")'''

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

    '''print("\nAnnual H₂ Production per Technology (kg H₂/year):")
    for year in sorted(D_y.keys()):
        print(f"Year {year}:")
        for tech in h2_output_per_tech:
            h2 = h2_output_per_tech[tech].get(year, 0)
            print(f"  {tech:<8}: {h2:,.2f} kg H₂")'''

    print("\nHydrogen Demand vs. Output:")
    for y in sorted(D_y.keys()):
        demand = D_y[y]
        output = h2_output_per_tech.get('PEM', {}).get(y, 0)
        print(f"Year {y}: Demand = {demand:,.2f} kg, Output = {output:,.2f} kg")

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

    # Save X_c for next comparison
    Xc_prev = Xc_full.copy()

    # === LCOH Update ===
    if capacities_changed:
        smoothed_LCOH = smooth_lcoh_update(initial_LCOH, updated_LCOH_raw, alpha=0.3)
    else:
        smoothed_LCOH = updated_LCOH_raw  # If stuck, use raw to allow shift

    # Add LCOE electricity cost based on energy tech used
    tech_to_source = {'PEM': 'Wind', 'Alkaline': 'PV'}
    prev_LCOH = initial_LCOH.copy()
    LCOE_LCOH = apply_lcoe_to_lcoh(smoothed_LCOH, tech_to_source, initial_LCOE, X_c, X_energy)
    initial_LCOH = smoothed_LCOH

    #LCOS = lcos_dynamic_q(total_stored)

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
            print("Convergence reached based on LCOH variation < threshold.")
            break


# Results & Plots.

'''#Costs
years = list(range(2025, 2051))
capex_per_year_plot = [CAPEX_info['PEM'].get(y, 0) + CAPEX_info['Alkaline'].get(y, 0) + CAPEX_info['SMR'].get(y, 0) + CAPEX_info['CCUS90'].get(y, 0)
                  for y in years]
opex_per_year_plot = [sum(opex_per_tech[tech].get(y, 0) for tech in opex_per_tech)
                 for y in years]
grid_opex_plot = [grid_opex_per_year.get(y, 0) for y in years]

fig = plt.figure(figsize=(12, 6))
plt.plot(years, capex_per_year_plot, label=['Total Discounted CAPEX'])
plt.plot(years, opex_per_year_plot,  label=['Total Discounted OPEX'])
plt.plot(years, grid_opex_plot, label=['Total Discounted Grid Costs'])
plt.title('Discounted Annual System Cost Breakdown')
plt.xlabel('Year')
plt.ylabel('Cost [USD]')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("cost_breakdown.png")
plt.show()

#Capex per tech
capex_df = pd.DataFrame(CAPEX_info).fillna(0).sort_index()  # Years as index, techs as columns

fig, ax = plt.subplots(figsize=(12, 6))
capex_df.plot(kind='bar', stacked=True, ax=ax)

ax.set_title('Annual CAPEX by Technology')
ax.set_xlabel('Year')
ax.set_ylabel('CAPEX (USD)')
ax.legend(title='Technology', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("capex_per_year_by_tech.png", dpi=300)
plt.show()

#LCOH evolution

# Sample data: replace with your actual LCOH tracking across iterations
lcoh_history = {
    'PEM': [4.3, 2.3293, 1.4841, 1.4841],
    'Alkaline': [3.25, 0.0, 0.0, 0.0],
    'SMR': [2.0, 5.8946, 7.5637, 7.5637],
    'CCUS90': [2.0, 0.0, 0.0, 0.0]
}

iterations = list(range(1, len(next(iter(lcoh_history.values()))) + 1))

plt.figure(figsize=(8, 5))
for tech, values in lcoh_history.items():
    plt.plot(iterations, values, marker='o', label=tech)

plt.title("LCOH Evolution Across Iterations")
plt.xlabel("Iteration")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel("LCOH (USD/kg H₂)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lcoh_evolution.png")
plt.show()

#Installed capacity per tech per year
years = sorted(next(iter(Xc_full.values())).keys())
techs = list(Xc_full.keys())

fig, ax = plt.subplots(figsize=(10, 6))

for tech in techs:
    values = [Xc_full[tech][y] for y in years]
    # Convert to consistent units
    ax.plot(years, values, label=tech)

ax.set_title('Installed Capacity per Technology in '+scenario_id+' (2025–2050)')
ax.set_xlabel("Year")
ax.set_ylabel("Installed Capacity (MW for Electrolysis, tH₂/year for Fossil)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('installed_capacity_trend'+scenario_id+'.png')
plt.show()

#Fossil vs renewable
# Categorize technologies
fossil_techs = ['SMR', 'CCUS90']
renewable_techs = ['PEM', 'Alkaline']
# Build time series data
fossil_output = [sum(h2_output_per_tech[t].get(y, 0) for t in fossil_techs) / 1e6 for y in years]
renewable_output = [sum(h2_output_per_tech[t].get(y, 0) for t in renewable_techs) / 1e6 for y in years]

plt.figure(figsize=(10, 6))
plt.stackplot(years, renewable_output, fossil_output, labels=["Renewable H₂", "Fossil H₂"])
plt.ylabel("Hydrogen Output (kT/year)")
plt.xlabel("Year")
plt.title("Fossil vs. Renewable Hydrogen Supply Over Time")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig("h2_output_fossil_vs_renewable.png")
plt.show()

# Hydrogen supply by technology
years = sorted(D_y.keys())
df_output = pd.DataFrame({
    tech: [h2_output_per_tech[tech].get(y, 0) / 1e6 for y in years]
    for tech in h2_output_per_tech
}, index=years)

plt.figure(figsize=(10, 6))
plt.stackplot(df_output.index, df_output.T, labels=df_output.columns)
plt.ylabel("Hydrogen Output (kT/year)")
plt.xlabel("Year")
plt.title("Annual Hydrogen Supply by Technology")
plt.legend(loc="upper left")
plt.grid(True, which='both', axis='y')
plt.tight_layout()
plt.savefig('h2_output_stacked_area'+scenario_id+'.png')
plt.show()


#Supply vs demand
total_supply = [
    sum(h2_output_per_tech[tech].get(y, 0) for tech in h2_output_per_tech) / 1e6
    for y in years
]
demand_series = [D_y[y] / 1e6 for y in years]  # convert to kT

plt.figure(figsize=(10, 6))
plt.plot(years, demand_series, label="Hydrogen Demand", linestyle="--", marker="o")
plt.plot(years, total_supply, label="Hydrogen Supply", linestyle="-", marker="s")
plt.title("Hydrogen Demand vs. Total Supply Over Time")
plt.ylabel("Hydrogen (kT/year)")
plt.xlabel("Year")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("h2_demand_vs_supply.png")
plt.show()

#storage

years = sorted(storage_utilization_dict.keys())
util_rates = [storage_utilization_dict[y] * 100 for y in years]  # Convert to %

plt.figure(figsize=(10, 6))
plt.plot(years, util_rates, marker='o', linewidth=2)
plt.title('Storage Utilization Rate Over Time')
plt.xlabel('Year')
plt.ylabel('Utilization Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('storage_utilization.png', dpi=300)
plt.show()

#Grid

years = list(range(2025, 2051))
grid_gwh = grid_import_year  # grid import in GWh per year
renewable_gwh = renewable_energy_yearly  # renewable supply in GWh per year

years = sorted(grid_gwh.keys() & renewable_gwh.keys())  # Ensure common sorted years
grid_share_pct = [100 * grid_gwh[y] / (grid_gwh[y] + renewable_gwh[y]) if (grid_gwh[y] + renewable_gwh[y]) > 0 else 0 for y in years]

plt.figure(figsize=(10, 6))
plt.plot(years, grid_share_pct, marker='o', linestyle='-', linewidth=2)
plt.title("Grid Electricity Share of Total Energy Supply for "+scenario_id)
plt.xlabel("Year")
plt.ylabel("Grid Share (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig('grid_share_over_time_for_'+scenario_id+'.png')
plt.show()


years = sorted(list(X_energy_per_year['PV'].keys()))
pv_cap = [X_energy_per_year['PV'][y] for y in years]
wind_cap = [X_energy_per_year['Wind'][y] for y in years]

plt.figure(figsize=(10, 6))
plt.plot(years, pv_cap, label='PV', marker='o', linewidth=2)
plt.plot(years, wind_cap, label='Wind', marker='s', linewidth=2)

plt.title('Installed Renewable Energy Capacity Over Time')
plt.xlabel('Year')
plt.ylabel('Installed Capacity (MW)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('renewable_capacity_trend_for_'+scenario_id+'.png')
plt.show()

# LCOH values for PEM and SMR in base and scenario 1
techs = ['PEM', 'SMR']
lcoh_base = [1.4841, 7.56]
lcoh_sc1 = [1.9823, 6.72]

bar_width = 0.35
x = range(len(techs))

plt.figure(figsize=(8, 5))
plt.bar([i - bar_width/2 for i in x], lcoh_base, width=bar_width, label='Base')
plt.bar([i + bar_width/2 for i in x], lcoh_sc1, width=bar_width, label='Scenario 1')

plt.xticks(x, techs)
plt.ylabel('LCOH (USD/kg H₂)')
plt.title('LCOH Comparison: Base vs '+scenario_id)
plt.legend()
plt.tight_layout()
plt.savefig('lcoh_comparison_scenario1.png')
plt.grid()
plt.show()


technologies = ['PEM', 'Alkaline', 'SMR', 'CCUS90']
lcoh_base = [1.48, 0.0, 7.56, 0.0]
lcoh_scenario3 = [22.33, 3.23, 1.85, 0.0]

x = np.arange(len(technologies))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar(x - width/2, lcoh_base, width, label='Base Scenario')
bar2 = ax.bar(x + width/2, lcoh_scenario3, width, label='Scenario 3')

ax.set_ylabel('LCOH (USD/kg H₂)')
ax.set_title('LCOH Comparison: Base vs. Scenario 3')
ax.set_xticks(x)
ax.set_xticklabels(technologies)
ax.legend()
plt.tight_layout()
plt.savefig('lcoh_comparison_base_vs_scenario3.png')
plt.show()

years = list(range(2025, 2051))
installed_storage_base = {2025: 0.0, 2026: 249.70974968547, 2027: 530.439295633791, 2028: 790.490168105206, 2029: 1073.45053759024, 2030: 1346.12192733248, 2031: 1541.23109726253, 2032: 1752.2058054704, 2033: 1963.18974053476, 2034: 2197.7045307181, 2035: 2460.92870908527, 2036: 2938.13507740981, 2037: 3033.74417928117, 2038: 3205.16212495693, 2039: 3438.43602164651, 2040: 3883.87723101287, 2041: 4769.86789629509, 2042: 5311.53921768426, 2043: 5621.23681152356, 2044: 6180.67486506548, 2045: 6873.78721526521, 2046: 7160.3331320134, 2047: 7796.46561068461, 2048: 8377.18013375122, 2049: 9104.40810642976, 2050: 9655.19478443341}       # base scenario installed capacity per year
installed_storage_scenario5 = Xstorage_test #{2025: 0.0, 2026: 491.791625015589, 2027: 1015.72078776831, 2028: 1368.20118787989, 2029: 2488.48019559201, 2030: 3307.50726278655, 2031: 3370.61264092338, 2032: 3717.43359613677, 2033: 5964.27293224627, 2034: 1598.11020572375, 2035: 5988.73169125329, 2036: 4004.88916454093, 2037: 14976.0982782748, 2038: 0.0, 2039: 4773.98854760187, 2040: 22274.0200762209, 2041: 19035.4697945674, 2042: 23027.6016363428, 2043: 0.0, 2044: 24382.5606785274, 2045: 3208.49574610955, 2046: 0.0, 2047: 0.0, 2048: 25502.8135771405, 2049: 0.0, 2050: 26310.8519015963}  # scenario 5 installed capacity per year
utilization_base = {2025: 0.0, 2026: 0.9374086734980364, 2027: 0.9392625670197343, 2028: 0.9324201423214209, 2029: 0.889136329535477, 2030: 0.8948291899177756, 2031: 0.8998913530451408, 2032: 0.9332716498578414, 2033: 0.9340828714643488, 2034: 0.9295140905760714, 2035: 0.8862628704264944, 2036: 0.8525525168208145, 2037: 0.8406239956104217, 2038: 0.8442617686824597, 2039: 0.874520254142183, 2040: 0.8425969918362413, 2041: 0.7988000965913963, 2042: 0.7678349955324051, 2043: 0.7909707166097741, 2044: 0.7657013770372398, 2045: 0.8051639438879862, 2046: 0.7793459370181779, 2047: 0.80045759673358, 2048: 0.8139026180410738, 2049: 0.8110444465753047, 2050: 0.7773170551311017} # utilization rate (%) per year in base
utilization_scenario5 = storage_utilization_dict#{2025: 0.0, 2026: 1.24933771670748, 2027: 0.8697851066164628, 2028: 0.987943703912691, 2029: 0.7297761952281155, 2030: 0.6708830316472716, 2031: 1.3380639335050473, 2032: 0.6043408586110836, 2033: 1.0025066095304755, 2034: 0.4248080600867416, 2035: 1.0351073252035037, 2036: 0.8332215002859275, 2037: 0.6281064894445859, 2038: 1.1189870858766349, 2039: 0.6547348711316149, 2040: 1.159494292464262, 2041: 0.31432669966008864, 2042: 0.3159482338197854, 2043: 0.0, 2044: 0.31007681319626235, 2045: 0.5823296932007457, 2046: 0.6799617351589811, 2047: 0.31013788638059286, 2048: 0.20534463461173194, 2049: 0.0, 2050: 0.797122411690898}       # utilization rate (%) per year in scenario 5
grid_import_scenario5 = grid_import_year#{2025: 0.0, 2026: 0.0, 2027: 0.0, 2028: 0.0, 2029: 0.0, 2030: 0.0, 2031: 0.0, 2032: 71.80157563354703, 2033: 158.44589473104904, 2034: 245.09021382855101, 2035: 331.73453292605296, 2036: 418.37885202355517, 2037: 505.0231711210571, 2038: 591.6674902185588, 2039: 678.3118093160608, 2040: 764.9561284135607, 2041: 949.2439608067928, 2042: 1133.5317932000244, 2043: 1317.8196255932567, 2044: 1675.0736009316468, 2045: 2089.3621218831704, 2046: 2550.081702866249, 2047: 3010.8012838493287, 2048: 3584.153548325694, 2049: 4229.160961702006, 2050: 4874.1683750783095}        # grid import (GWh) per year in scenario 5

# Ensure values are aligned by year
installed_storage_base_values = [installed_storage_base[y] for y in years]
installed_storage_scenario5_values = [installed_storage_scenario5[y] for y in years]
utilization_base_values = [utilization_base[y] * 100 for y in years]  # convert to %
utilization_scenario5_values = [utilization_scenario5[y] * 100 for y in years]  # convert to %
grid_import_values = [grid_import_scenario5[y] for y in years]

# 1. Line Plot – Installed Storage Capacity Comparison
plt.figure()
plt.plot(years, installed_storage_base_values, label='Base Scenario')
plt.plot(years, installed_storage_scenario5_values, label='Scenario 5')
plt.xlabel('Year')
plt.ylabel('Installed Storage Capacity [kg]')
plt.title('Storage Capacity Installed Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('storage_installed_comparison.png')
plt.show()

# 2. Line Plot – Storage Utilization Comparison
plt.figure()
plt.plot(years, utilization_base_values, label='Base Scenario')
plt.plot(years, utilization_scenario5_values, label='Scenario 5')
plt.xlabel('Year')
plt.ylabel('Utilization Rate [%]')
plt.title('Storage Utilization Rate Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('storage_utilization_comparison.png')
plt.show()

# 3. Bar Plot – Grid Import Over Time (Scenario 5)
plt.figure()
plt.bar(years, grid_import_values)
plt.xlabel('Year')
plt.ylabel('Grid Import [GWh]')
plt.title('Annual Grid Imports – Scenario 5')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('grid_import_scenario5.png')
plt.show()'''










