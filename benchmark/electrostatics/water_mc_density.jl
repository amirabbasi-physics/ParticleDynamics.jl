# Ensemble check: SPC/E water density under Monte Carlo volume moves
# (molecular scaling, every MC_INTERVAL steps) with CSVR at 300 K and 1 bar, on the
# device engine, from a GROMACS-equilibrated water box (Float32, cluster backend).
# Compared with the GROMACS C-rescale reference (water_gromacs_npt.sh). The MC step
# size is tuned during the first half (equilibration) and frozen for production.
# VDW_MODIFIER=potential_shift matches GROMACS' default LJ convention (default: none,
# plain truncation; compare that with GROMACS at 1 + 99.7 bar, see water_gromacs_npt.sh).
# BAROSTAT=crescale uses molecular C-rescale with GROMACS' reference
# settings (tau_p = 1 ps, compressibility 4.5e-5 /bar, an event every 20 steps) instead;
# BAROSTAT=crescale_device runs the same C-rescale inside the device engine's step graphs
# (device-resident barostat plan B3) and samples the box at a host boundary every 100 steps.
# Usage: julia --project=. water_mc_density.jl SYSTEMS_DIR NAME OUT.toml [NSTEPS]
include(joinpath(@__DIR__, "rigid_water_states.jl"))
sysdir = ARGS[1]; name = ARGS[2]; out = ARGS[3]
nsteps = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 250_000
const BAROSTAT = Symbol(get(ENV, "BAROSTAT", "mc"))
const MC_INTERVAL = BAROSTAT in (:crescale, :crescale_device) ? 20 : 25
const KB = 0.0083144626
const BAR = 0.0602214076            # kJ/mol/nm³
const MW = 15.9994 + 2 * 1.008      # g/mol
st, d = load_rigid_water(sysdir, name)
const MODIFIER = Symbol(get(ENV, "VDW_MODIFIER", "none"))
set_vdw_modifier!(st, MODIFIER)
nw = length(st.rx) ÷ 3
density(st) = nw * MW / (602.214076 * cell_volume(st))      # g/cm³
kT = KB * 300
sp = SC.csvr(st; temperature=Float32(kT), tau=Float32(0.1))
attempts = nsteps ÷ MC_INTERVAL
mc = BAROSTAT in (:crescale, :crescale_device) ?
    SC.StochasticCellRescalingState(st; pressure=BAR, kT, compressibility=4.5e-5 / BAR, tau=1.0, seed=0x3b) :
    SC.MonteCarloVolumeMove(st; pressure=BAR, kT, log_volume_step=0.005, seed=0x3b, tune_batches=attempts ÷ 2 ÷ 100)
pressure_event!(st, mc::SC.MonteCarloVolumeMove) = SC.monte_carlo_volume_move!(st, mc)
pressure_event!(st, mc::SC.StochasticCellRescalingState) =
    SC.stochastic_cell_rescale!(st, mc, MC_INTERVAL * st.dt; compute_energy=false)
acceptance(mc::SC.MonteCarloVolumeMove) = SC.acceptance_ratio(mc)
acceptance(mc) = NaN
step_size(mc::SC.MonteCarloVolumeMove) = mc.log_volume_step
step_size(mc) = NaN
ρ = Float64[]; t0 = time()
BAROSTAT === :crescale_device && SC.attach_pressure_control!(st, mc; interval=MC_INTERVAL)
for i in 1:nsteps
    if BAROSTAT === :crescale_device
        SC.step!(st, sp, st.dt; compute_energy=false)
        i % 100 == 0 && (DE.synchronize_state!(st); push!(ρ, density(st)))
    else
        SC.step!(st, sp, st.dt; compute_energy=i % MC_INTERVAL == 0)
        if i % MC_INTERVAL == 0
            pressure_event!(st, mc)
            push!(ρ, density(st))
        end
    end
    if i % 25_000 == 0
        @printf("step %7d  ρ = %.5f g/cm³  acceptance %.3f  δ = %.5f  (%.0f s)\n", i, ρ[end],
                acceptance(mc), step_size(mc), time() - t0); flush(stdout)
    end
end
DE.synchronize_state!(st)
st.engine isa DE.StateEngine || error("engine not used")
dl, dv = constraint_errors(st)
prod_ρ = ρ[end÷2+1:end]
nb = 10; bl = length(prod_ρ) ÷ nb
blocks = [sum(prod_ρ[(k-1)*bl+1:k*bl]) / bl for k in 1:nb]
m = sum(blocks) / nb; se = sqrt(sum(abs2, blocks .- m) / (nb * (nb - 1)))
@printf("%s (barostat=%s, vdw_modifier=%s): ρ = %.5f ± %.5f g/cm³ (production %d samples), acceptance %.3f, constraint errors %.1e / %.1e\n",
        name, BAROSTAT, MODIFIER, m, se, length(prod_ρ), acceptance(mc), dl, dv)
open(io -> TOML.print(io, Dict("name" => name, "nsteps" => nsteps, "vdw_modifier" => String(MODIFIER), "mc_interval" => MC_INTERVAL, "barostat" => String(BAROSTAT),
    "density_mean" => m, "density_stderr" => se, "density_trace" => ρ, "acceptance" => acceptance(mc),
    "log_volume_step" => step_size(mc), "constraint_length_error" => dl, "constraint_velocity_error" => dv,
    "start_density" => nw * MW / (602.214076 * prod(Float64.(d.f.box))))), out, "w")
