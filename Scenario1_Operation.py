from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, NonNegativeReals,
SolverFactory, Param, Set, RangeSet, minimize, value)
solverpath_exe='/usr/local/bin/glpsol'

def run_operation(X_c, X_energy, X_storage, X_water, D_y, CAPEX_per_year):
    opex_per_tech = {c: {} for c in ['PEM', 'Alkaline', 'SMR', 'CCUS90']}
    h2_output_per_tech = {c: {} for c in ['PEM', 'Alkaline', 'SMR', 'CCUS90']}
    total_opex_all_years = 0
    grid_opex_per_year = {}
    total_discounted_opex = 0
    discount_rate = 0.1
    grid_import_year = {}
    renewable_energy_yearly = {}

    for year in sorted(D_y.keys()):
        model = ConcreteModel()
        model.T = RangeSet(1, 168)
        model.C = Set(initialize=list(X_c.keys()))
        model.R = Set(initialize=list(X_energy.keys()))

        model.X_c = Param(model.C, initialize=X_c)
        model.X_energy = Param(model.R, initialize=X_energy)
        model.X_water = Param(initialize=X_water)
        model.X_storage = Param(initialize=X_storage)
        model.E_H2 = Param(initialize=0.05)
        model.W_H2 = Param(initialize=0.009)
        model.hours_per_year = Param(initialize=8760)

        hourly_cf = {
            'PV': [0, 0, 0, 0, 0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.7, 0.6, 0.3, 0.1, 0, 0, 0, 0],
            'Wind': [0, 0, 0.6, 0.7, 0.8, 0.9, 0.9, 0.85, 0.75, 0.6, 0.5, 0.4, 0.3, 0.25, 0.25, 0.4, 0.5, 0.65, 0.8, 0.9, 0.8, 0.9, 0.9, 0.8]
        }

        model.lambda_r = Param(model.T, model.R, initialize=lambda model, t, r: hourly_cf[r][t % 24])
        model.RA = Param(model.T, model.R, initialize=lambda model, t, r: model.lambda_r[t, r] * model.X_energy[r])
        model.C_grid = Param(initialize=160) #USD/MWh
        opex_pct = {'PEM': 0.03, 'Alkaline': 0.03, 'SMR': 0.01, 'CCUS90': 0.025}
        C_c = {c: (opex_pct[c] * max(0, CAPEX_per_year[c].get(year, 0))) / (X_c[c] * 8760) if X_c[c] > 0 else 0 for c in X_c}
        model.C_c = Param(model.C, initialize=C_c)
        model.S_0 = Param(initialize=0)  # Initial storage level (kg H2)
        model.eta = Param(model.C, initialize={'PEM': 0.85, 'Alkaline': 0.55, 'SMR': 1, 'CCUS90': 1})
        model.D_t = Param(model.T, initialize=lambda model, t: D_y[year] / model.hours_per_year)

        #Decision variables
        model.E = Var(model.T, model.C, domain=NonNegativeReals, initialize=0)
        model.E_t = Var(model.T, model.R, domain=NonNegativeReals, initialize=0)
        model.I_grid = Var(model.T, domain=NonNegativeReals, initialize=0)
        model.H_t = Var(model.T, domain=NonNegativeReals, initialize=0) # Electrolysis hydrogen (kg)
        model.H_fossil = Var(model.T, model.C, domain=NonNegativeReals, initialize=0)  # Fossil hydrogen (kg)
        model.S_level = Var(model.T, domain=NonNegativeReals, initialize=0)  # Storage level (kg H2)
        model.S_in = Var(model.T, domain=NonNegativeReals, initialize=0)  # Hydrogen stored (kg H2)
        model.S_out = Var(model.T, domain=NonNegativeReals, initialize=0)  # Hydrogen released from storage (kg H2)

        def operational_cost_rule(model):
            energy_costs = sum(model.C_grid * model.I_grid[t] for t in model.T)
            conversion_costs = sum(model.C_c[c] * model.E[t, c] for c in ['PEM', 'Alkaline'] for t in model.T)
            fossil_costs = sum(model.H_fossil[t, c] * model.C_c[c] for c in ['SMR', 'CCUS90']for t in model.T)
            return energy_costs + conversion_costs + fossil_costs
        model.OPEX = Objective(rule=operational_cost_rule, sense=minimize)

        # 1. Energy Balance: Renewable + Grid Must Meet Demand, grid activation constraint
        def energy_balance_rule(model, t):
            return sum(model.E[t, c] for c in model.C) == sum(model.E_t[t, r]  for r in model.R) + model.I_grid[t]

        model.EnergyBalance = Constraint(model.T, rule=energy_balance_rule)

        # 2. Energy Production Limit from renewables
        def energy_production_rule(model, t, r):
            return model.E_t[t, r] <= model.RA[t, r]
        model.EnergyProduction = Constraint(model.T, model.R, rule=energy_production_rule)

        # 3. Conversion Capacity Limit (Fix)
        def conversion_capacity_limit_rule(model, t, c):
            return model.E[t, c] <= model.X_c[c]  # ∆t = 1 hour
        model.ConversionCapacityLimit = Constraint(model.T, model.C, rule=conversion_capacity_limit_rule)

        # 4. Storage Balance
        def storage_balance_rule(model, t):
            if t > 1:
                return model.S_level[t] == model.S_level[t - 1] + model.S_in[t] - model.S_out[t]
            else:
                return model.S_level[t] == model.S_0 + model.S_in[t] - model.S_out[t]

        model.StorageBalance = Constraint(model.T, rule=storage_balance_rule)

        # Storage Constraints
        model.StorageExtraction = Constraint(model.T, rule=lambda model, t: model.S_level[t] >= model.S_out[t])
        model.StorageInflow = Constraint(model.T, rule=lambda model, t: model.S_in[t] == model.H_t[t] - model.D_t[t])
        model.StorageCapacity = Constraint(model.T, rule=lambda model, t: model.S_level[t] <= model.X_storage)
        model.StorageUsage = Constraint(model.T, rule=lambda model, t: model.S_out[t] >= model.D_t[t] - model.H_t[t])

        # 5. Hydrogen production constraint
        def hydrogen_production_rule(model, t):
            return model.H_t[t] == sum(model.E[t, c] * model.eta[c] / model.E_H2 for c in ['PEM', 'Alkaline']) + \
                           sum(model.H_fossil[t, c] for c in ['SMR', 'CCUS90'])
        model.HydrogenProduction = Constraint(model.T, rule=hydrogen_production_rule)

        # 6. Hydrogen Demand Satisfaction
        def hydrogen_demand_satisfaction_rule(model, t):
            return model.H_t[t] >= model.D_t[t] # Ensure we meet demand in every hour and store 5% (*1.05?)
        model.HydrogenDemandSatisfaction = Constraint(model.T, rule=hydrogen_demand_satisfaction_rule)

        # 7. Gray hydrogen Demand Satisfaction
        def fossil_capacity_limit_rule(model, t, c):
            if c in ['SMR', 'CCUS90']:
                # Max fossil production per hour = capacity * 1 hour
                return model.H_fossil[t, c] <= model.X_c[c]
            else:
                return Constraint.Skip
        model.FossilCapacityLimit = Constraint(model.T, model.C, rule=fossil_capacity_limit_rule)

        solver = SolverFactory('glpk', executable=solverpath_exe)
        result = solver.solve(model, tee=False)

        '''print("\nHydrogen Demand Satisfaction Check (H_t vs D_t):")
        for t in model.T:
            if model.H_t[t].value is not None:
                print(f" Year {year}:, t={t:>3}: Demand={model.D_t[t]:,.2f}, H_t={model.H_t[t].value:,.2f}")'''

        '''print("\nE[t,c] energy use:")
        for t in model.T:
            for c in model.C:
                if model.E[t, c].value and model.E[t, c].value > 0:
                    print(f"t={t}, c={c}, E={model.E[t, c].value:.4f}")'''

        if result.solver.termination_condition == 'optimal':
            # Conversion Tech OPEX (per tech)
            tech_opex = {
                c: sum(value(model.E[t, c]) * value(model.C_c[c]) for t in model.T)
                for c in model.C if c in ['PEM', 'Alkaline']
            }

            # Fossil Tech OPEX (per tech, using H_fossil)
            fossil_opex = {
                c: sum(value(model.H_fossil[t, c]) * value(model.C_c[c]) for t in model.T)
            for c in ['SMR', 'CCUS90']
            }

            # Grid import OPEX
            grid_opex = sum(value(model.I_grid[t]) * value(model.C_grid) for t in model.T)
            grid_opex_per_year[year] = grid_opex * 52

            for c in model.C:
                if c in ['PEM', 'Alkaline']:
                    total_h2 = sum(value(model.E[t, c]) * value(model.eta[c]) / value(model.E_H2) for t in model.T)
                elif c in ['SMR', 'CCUS90']:
                    total_h2 = sum(value(model.H_fossil[t, c]) for t in model.T)
                else:
                    total_h2 = 0
                h2_output_per_tech[c][year] = total_h2 * 52

            # Accumulate total
            total_opex = (sum(tech_opex.values()) + sum(fossil_opex.values()) + grid_opex) * 52
            total_opex_all_years += total_opex
            raw_opex = sum(tech_opex.values()) + sum(fossil_opex.values()) + grid_opex
            discount_factor = 1 / ((1 + discount_rate) ** (year - 2025))
            discounted_opex = raw_opex * 52 * discount_factor
            total_discounted_opex += discounted_opex

            # Store OPEX per tech per year
            for c in model.C:
                if c in tech_opex:
                    opex_per_tech[c][year] = tech_opex[c] * 52 # Annualized
                elif c in fossil_opex:
                    opex_per_tech[c][year] = fossil_opex[c] * 52 # Annualized
                else:
                    opex_per_tech[c][year] = 0  # In case no value was recorded

            # Grid import OPEX
            grid_import_year[year] = sum(value(model.I_grid[t]) for t in model.T) * 52 / 1000
            print(f"Grid import for year {year}: {sum(value(model.I_grid[t]) for t in model.T)* 52:,.2f} MWh-año")
            #print(f"Grid OPEX for year {year}: {grid_opex_per_year[year] :,.2f} USD")

            total_renewable_energy = sum(
                value(model.E_t[t, r]) for t in model.T for r in model.R) * 52 / 1000  # Convert to GWh
            renewable_energy_yearly[year] = total_renewable_energy

    print("\nSolver Termination:", result.solver.termination_condition)

    return total_discounted_opex, opex_per_tech, h2_output_per_tech, grid_opex_per_year, grid_import_year, renewable_energy_yearly
