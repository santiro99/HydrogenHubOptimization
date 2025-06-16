from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, SolverFactory, Param, Set, RangeSet, minimize, value, Expression
solverpath_exe='/usr/local/bin/glpsol'
import random
import math

def run_planning(LCOH_c, LCOE_r, LCOS, Xc_prev=None):
    model = ConcreteModel()

    if Xc_prev is None:
        Xc_prev = {c: 0 for c in ['PEM', 'Alkaline', 'SMR', 'CCUS90']}

    model.Y = RangeSet(2025, 2050)
    model.C = Set(initialize=['PEM', 'Alkaline', 'SMR', 'CCUS90'])
    model.R = Set(initialize=['PV', 'Wind', 'Purified Water'])

    model.E_H2 = Param(initialize=0.045) #Energy used per kg of H2
    model.eta = Param(model.C, initialize={'PEM': 0.75, 'Alkaline': 0.55, 'SMR': 0.83, 'CCUS90': 0.7}) #Efficiency
    model.phi1 = Param(initialize=0.07)  # Discount rate
    model.hours_per_year = Param(initialize=8760)
    grid_share_limit = 0.2  # Allow max from grid in energy balance planning.
    CF_r = {'PV': 0.16, 'Wind': 0.36}  # Irena average values

    model.emissions = Param(model.C, initialize={'PEM': 0, 'Alkaline': 0, 'SMR': 7, 'CCUS90': 3})
    model.LCOH = Param(model.C, initialize=LCOH_c)
    model.LCOE = Param(model.R, initialize=LCOE_r)
    model.LCOS = Param(initialize=LCOS)

    def fossil_share_function(y, start_year=2025, end_year=2051, initial_share=0.95, final_share=0.0):
        """Returns the fossil share in year y, decreasing exponentially from initial_share to final_share."""
        decay_rate = math.log(initial_share / (final_share + 1e-4)) / (end_year - start_year)
        return max(final_share, initial_share * math.exp(-decay_rate * (y - start_year)))

    fossil_share_data = {y: fossil_share_function(y) for y in range(2025, 2051)}
    model.fossil_share = Param(model.Y, initialize=fossil_share_data)

    storage_factor = {
        y: random.uniform(1.2e-5, 1.3e-5) if y <= 2030 else
        random.uniform(1.2e-5, 1.3e-5) if y <= 2035 else
        random.uniform(1.3e-5, 1.4e-5) if y <= 2040 else
        random.uniform(1.4e-5, 1.45e-5) if y <= 2045 else
        random.uniform(1.45e-5, 1.5e-5)
        for y in range(2025, 2051)
    }
    model.storage_factor = Param(model.Y, initialize=storage_factor)

    def interpolate_demand(y):
        demand_values = {2030: 103780000, 2040: 281020000, 2050: 658000000}
        if y < 2030:
            return demand_values[2030] * (y - 2025) / 5
        elif y < 2040:
            return demand_values[2030] + (demand_values[2040] - demand_values[2030]) * (y - 2030) / 10
        else:
            return demand_values[2040] + (demand_values[2050] - demand_values[2040]) * (y - 2040) / 10

    model.h2_demand = Param(model.Y, initialize=lambda m, y: interpolate_demand(y))

    # CAPEX for conversion technologies USD/MW
    model.CAPEX_c = Param(model.C, model.Y, initialize=lambda model, c, y: (
    {('PEM', 2025): 1250000, ('PEM', 2050): 200000, ('Alkaline', 2025): 500000, ('Alkaline', 2050): 150000}
    [c, y] if c in ['PEM', 'Alkaline'] and y in {2025, 2050} else
    {'PEM': 1250000, 'Alkaline': 500000}[c] + (y - 2025) * ({'PEM': (200000 - 1250000), 'Alkaline': (150000 - 500000)}[c] / 25)
    if c in ['PEM', 'Alkaline']
    else
    {('SMR', 2025): 80000, ('SMR', 2050): 70000, ('CCUS90', 2025): 140000, ('CCUS90', 2050): 140000}
    [c, y] if c in ['SMR', 'CCUS90'] and y in {2025, 2050} else
    (80000 + (y - 2025) * ((70000 - 80000) / 25) if c == 'SMR' else 140000)
    ))

    # CAPEX for resources
    model.CAPEX_r = Param(model.R, model.Y, initialize=lambda model, r, y: (
    {('PV', 2025): 200, ('PV', 2050): 100, ('Wind', 2025): 400, ('Wind', 2050): 250,
    ('Purified Water', 2025): 1.2, ('Purified Water', 2050): 1}
    [r, y] if r in model.R and y in {2025, 2050} else
    {'PV': 300, 'Wind': 400, 'Purified Water': 1.2}[r] +
    (y - 2025) * (({'PV': (150 - 300), 'Wind': (250 - 400), 'Purified Water': (1 - 1.2)}[r]) / 25)
    ))

    #Decision variables

    model.X_c = Var(model.C, model.Y, domain=NonNegativeReals)
    model.X_energy = Var(['PV', 'Wind'], model.Y, domain=NonNegativeReals)
    model.X_water = Var(model.Y, domain=NonNegativeReals)
    model.X_storage = Var(model.Y, domain=NonNegativeReals)
    model.H_annual = Var(model.Y, domain=NonNegativeReals)

    #Penalty variable for smoothness
    model.Dev = Var(model.C, model.Y, domain=NonNegativeReals)
    def dev_abs_rule_pos(model, c, y):
        return model.Dev[c, y] >= model.X_c[c, y] - Xc_prev.get((c, y), 0)
    model.DevPos = Constraint(model.C, model.Y, rule=dev_abs_rule_pos)
    def dev_abs_rule_neg(model, c, y):
        return model.Dev[c, y] >= Xc_prev.get((c, y), 0) - model.X_c[c, y]
    model.DevNeg = Constraint(model.C, model.Y, rule=dev_abs_rule_neg)

    def hydrogen_demand_rule(model, y):
        return model.H_annual[y] >= model.h2_demand[y] + model.X_storage[y]
    model.HydrogenDemand = Constraint(model.Y, rule=hydrogen_demand_rule)

    def conversion_capacity_rule(model, y):
        return sum(model.X_c[c, y] * model.eta[c] * model.hours_per_year for c in ['PEM', 'Alkaline']) >= model.H_annual[y] * (1 - model.fossil_share[y]) * model.E_H2
    model.CapacityDemand = Constraint(model.Y, rule=conversion_capacity_rule)

    def conversion_capacity_fossil_rule(model, y):
        return sum(model.X_c[c, y] * model.eta[c] * model.hours_per_year for c in ['SMR', 'CCUS90']) \
               >= model.H_annual[y] * model.fossil_share[y]
    model.CapacityDemandFossil = Constraint(model.Y, rule=conversion_capacity_fossil_rule)

    def renewable_share_rule(model, y):
        renewable_energy = sum(model.X_energy[r, y] * CF_r[r] * model.hours_per_year for r in ['PV', 'Wind'])
        total_required_energy = model.H_annual[y] * model.E_H2
        return renewable_energy >= (1 - grid_share_limit) * total_required_energy
    model.RenewableShare = Constraint(model.Y, rule=renewable_share_rule)

    def storage_capacity_rule(model, y):
        return model.X_storage[y] >= model.storage_factor[y] * model.H_annual[y]
    model.Storage = Constraint(model.Y, rule=storage_capacity_rule)

    def water_sufficiency_rule(model, y):
        return model.X_water[y] >= model.H_annual[y] * 0.009
    model.Water = Constraint(model.Y, rule=water_sufficiency_rule)

    # Objective function
    def total_cost_rule(model):
        penalty_weight = 10000
        model.Penalty = Expression(expr=sum(model.Dev[c, y] for c in model.C for y in model.Y)) # Penalty for smoothness of changes between iterations.
        conv_cost = sum(model.X_c[c, y] * model.LCOH[c] for c in model.C for y in model.Y)
        energy_cost = sum(model.X_energy[r, y] * model.LCOE[r] for r in ['PV', 'Wind'] for y in model.Y)
        storage_cost = sum(model.X_storage[y] * model.LCOS for y in model.Y)
        emission_cost = sum(model.X_c[c, y] * model.emissions[c] for c in model.C for y in model.Y)
        return conv_cost + energy_cost + storage_cost + emission_cost + penalty_weight * model.Penalty

    model.TotalCost = Objective(rule=total_cost_rule, sense=minimize)

    solver = SolverFactory('glpk', executable=solverpath_exe)
    solver.solve(model)

    # Return output values for operation
    last = max(model.Y)

    Xc = {
        c: max(value(model.X_c[c, y]) for y in model.Y)
        if any(value(model.X_c[c, y]) is not None for y in model.Y) else 0.0
        for c in model.C
    }

    Xr = {r: value(model.X_energy[r, last]) if model.X_energy[r, last].value is not None else 0.0 for r in ['PV', 'Wind']}
    Xs = value(model.X_storage[last] if model.X_storage[last].value is not None else 0.0)
    #Xstorage_test = {y: value(model.X_storage[y]) for y in model.Y}
    Xw = value(model.X_water[last] if model.X_water[last].value is not None else 0.0)
    Dy = {y: value(model.h2_demand[y]) for y in model.Y}
    full_Xc_series = {c: {y: value(model.X_c[c, y]) for y in model.Y} for c in model.C}

    X_energy_per_year = {r: {y: value(model.X_energy[r, y]) for y in model.Y} for r in ['PV', 'Wind']}

    CAPEX_info = {}
    CAPEX_per_year = {c: {} for c in model.C}

    for c in model.C:
        capex_total = 0
        for y in model.Y:
            added_capacity = max(0, value(model.X_c[c, y]) - (value(model.X_c[c, y - 1]) if y > 2025 else 0))
            added_capex = value(model.CAPEX_c[c, y]) * added_capacity
            CAPEX_per_year[c][y] = added_capex  # Store non-discounted per year
            capex_total += added_capex  # Sum for total
        CAPEX_info[c] = capex_total

    CAPEX_info_renw = {}
    CAPEX_per_year_renw = {r: {} for r in model.R}

    for r in ['PV', 'Wind']:
        capex_total = 0
        for y in model.Y:
            added_capacity = value(model.X_energy[r, y]) - (value(model.X_energy[r, y - 1]) if y > 2025 else 0)
            added_capex = value(model.CAPEX_r[r, y]) * added_capacity
            CAPEX_per_year_renw[r][y] = added_capex
            capex_total += added_capex
        CAPEX_info_renw[r] = capex_total

    '''print("\nInstalled Renewable Energy Capacities (MW):")
    for r in ['PV', 'Wind']:
        for y in model.Y:
            cap = value(model.X_energy[r, y])
            print(f"{r} ({y}): {cap:.2f} MW")

    print("\nInstalled Conversion  Capacities (MW):")
    for c in model.C:
        for y in model.Y:
            cap = value(model.X_c[c, y])
            print(f"{c} ({y}): {cap:.2f} MW")

    print("\nInstalled Storage Capacity (Planning):")
    for y in model.Y:
        storage_cap = value(model.X_storage[y])
        print(f"  Year {y}: {storage_cap:,.2f} kg")'''

    #return Xc, Xr, Xs, Xstorage_test, Xw, Dy, CAPEX_per_year, CAPEX_per_year_renw, full_Xc_series
    return Xc, Xr, Xs, Xw, Dy, CAPEX_per_year, CAPEX_per_year_renw, full_Xc_series, X_energy_per_year


