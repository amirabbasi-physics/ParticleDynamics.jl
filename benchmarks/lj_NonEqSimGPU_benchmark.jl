import NonEqSimGPU

function simulation_submit(; num_runs::Int, Npart::Int, homogeneity::Union{Bool,String}, collision::Bool, phi::T, alpha_2::T, cold_phi::T, force_func::Function) where T
        @time NonEqSimGPU.sim_run(
            restart         =   nothing,
            type            =   "Langevin",
            num_runs        =   num_runs,
            homogeneous     =   homogeneity,
            collision_calc  =   collision,
            num_steps       =   Int(1e5),
            save_interval   =   Int(1e4),
            Npart           =   Npart,
            ptypes          =   ["C","C"],
            p_ids           =   [0, 0],
            dim             =   2,
            ϕ               =   phi,
            fraction        =   0.5f0,
            cold_frac       =   cold_phi,
            R               =   Float32(3.405e-10),
            ϵ               =   Float32(1e8),
            neigh_cut_off   =   4.0f0,
            neigh_update    =   10000,
            α₁              =   alpha_2,
            α₂              =   alpha_2,
            density         =   402.7f0,
            η               =   Float32(0.00234),
            Δt_prod         =   Float32(2e-6),
            integ           =   "vv", 
            random_positions      = false,
            force_func      = force_func,
            relax_steps    =   Int(0),
            relax_temp   = alpha_2)
end

function run()
    homogs = [true]
    for homo in homogs
        num_runs    =   1
        Npart       =   10000
        phi         =   0.3f0
        alpha_2     =   100.0f0
        homogeneity =   homo
        collision   =   false
        cold_phi    =   1.0f0
        force_func  = NonEqSimGPU.Harmonic
        simulation_submit(num_runs = num_runs, Npart = Npart, homogeneity = homogeneity, phi = phi, collision = collision, cold_phi = cold_phi, alpha_2 = alpha_2, force_func = force_func)
    end
end

run()