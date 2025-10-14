#!/sw/spack-levante/julia-1.7.0-wm6d6v/bin/julia

#SBATCH --job-name=...
#SBATCH --partition=compute
#SBATCH --account=...
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=128  # Request all CPUs on the node
#SBATCH --time=08:00:00  # Adjust based on your expected runtime
#SBATCH --output=julia-%j.out
#SBATCH --error=julia-%j.err

using Distributed
using ClusterManagers

# Add workers
addprocs_slurm(Int(128*20), nodes=20, partition="compute")
println("Number of workers: $(Distributed.nworkers())")
println("workers added")
flush(stdout)

@everywhere begin
    using ProgressMeter
    using SpecialFunctions
    using LinearAlgebra
    using NaNStatistics    
    using Roots
    using NPZ

    @inline function omega2_true(kx, ky, kz, s, f0, N0)
        """
        Computes the true frequency squared (omega.^2) for a given set of wavenumbers (kx, ky, kz),
        stratification parameter (s), Coriolis parameter (f0), and buoyancy frequency (N0).
        Args:
            kx (float): Wavenumber in the x-direction.
            ky (float): Wavenumber in the y-direction.
            kz (float): Wavenumber in the z-direction.
            s (float): Stratification parameter.
            f0 (float): Coriolis parameter.
            N0 (float): Buoyancy frequency.
        Returns:
            float: The true frequency squared (omega.^2).
        """
        k2 = kx.^2 .+ ky.^2 .+ kz.^2
        ksqr = kx.^2 .+ ky.^2
        om2 = ifelse.(k2 .> 0, (N0.^2 .* ksqr .+ f0.^2 .* kz.^2) ./ k2, 0.0)
        return om2
    end

    @inline function coeff_p(kx, ky, kz,om, s, f0, N0)
        """
        Computes the coefficient 'p' for a given set of wavenumbers (kx, ky, kz),
        stratification parameter (s), Coriolis parameter (f0), and buoyancy frequency (N0).
        Args:
            kx (float): Wavenumber in the x-direction.
            ky (float): Wavenumber in the y-direction.
            kz (float): Wavenumber in the z-direction.
            s (float): Stratification parameter.
            f0 (float): Coriolis parameter.
            N0 (float): Buoyancy frequency.
        Returns:
            float: The coefficient 'p'.
        """
        k2 = kx.^2 .+ ky.^2 .+ kz.^2
        ksqr = kx.^2 .+ ky.^2
        om2 = om.^2
        if s == 0
            p = (N0.^2 .* f0.^2) ./ (N0.^2 .* ksqr .+ f0.^2 .* kz.^2)
        else
            p = ifelse.(k2 .== 0, 0, abs.((N0.^2 .- s.^2 .* om2) .* (s.^2 .* om2 .- f0.^2)) ./ ((1.0 .+ s.^2) .* k2 .* om2))
            p = ifelse.(ksqr .== 0 .&& abs.(kz) .> 0, 0.5, p)
            p = ifelse.(ksqr .> 0 .&& kz .== 0, N0^2 / 2.0,p)
        end
        return p
    end

    @inline function vec_q(kx, ky, kz,om, s, f0, N0)
        """
        Calculate the components of the wave vector for a given set of inputs.
        Parameters:
        - kx: Float64, wavenumber in x-direction
        - ky: Float64, wavenumber in y-direction
        - kz: Float64, wavenumber in z-direction
        - s: Int, parameter
        - f0: Float64, parameter
        - N0: Float64, parameter

        Returns:
        - qx: Complex{Float64}, wave vector component in x-direction
        - qy: Complex{Float64}, wave vector component in y-direction
        - qz: Complex{Float64}, wave vector component in z-direction
        - qb: Complex{Float64}, wave vector component b
        """
        
        icmplx = complex(0, 1)
        k2 = kx.^2 .+ ky.^2 .+ kz.^2
        ksqr = kx.^2 .+ ky.^2
        om2 = om.^2

        if s == 0
            division1 = f0.^2
            division2 = N0.^2
            qx = (.-icmplx .* f0 .* ky) ./ division1
            qy = (icmplx .* f0 .* kx) ./ division1
            qz = 0
            qb = icmplx .* kz .* N0.^2 ./ division2
        else
            division1 = ifelse.(ksqr .== 0, 0, f0.^2 .- s.^2 .* om2)
            division2 = ifelse.(kz .== 0, 0, N0.^2 .- s.^2 .* om2)
            qx = ifelse.(ksqr .== 0 .&& kz.^2 .> 0, .-icmplx .* s, (.-icmplx .* f0 .* ky .+ s .* om .* kx) ./ division1)
            qy = ifelse.(ksqr .== 0 .&& kz.^2 .> 0, s.^2, (icmplx .* f0 .* kx .+ s .* om .* ky) ./ division1)
            qz = ifelse.(ksqr .== 0 .&& kz.^2 .> 0, 0, kz .* s .* om ./ division2)
            qb = ifelse.(ksqr .== 0 .&& kz.^2 .> 0, 0, icmplx .* kz .* N0.^2 ./ division2)

            qx = ifelse.(ksqr .> 0 .&& kz.^2 .== 0, 0.0, qx)
            qy = ifelse.(ksqr .> 0 .&& kz.^2 .== 0, 0.0, qy)
            qz = ifelse.(ksqr .> 0 .&& kz.^2 .== 0, -icmplx .* s/N0, qz)
            qb = ifelse.(ksqr .> 0 .&& kz.^2 .== 0, 1.0, qb)
        end
        return (qx, qy, qz, qb)
    end

    @inline function vec_p(kx, ky, kz,om, s, f0, N0)
        om2 = om.^2
        qx, qy, qz, qb = vec_q.(kx, ky, kz,om, s, f0, N0)
        p = coeff_p.(kx, ky, kz,om, s, f0, N0)
        ksqr = kx.^2 .+ ky.^2
        qx_conj = conj.(qx)
        qy_conj = conj.(qy)
        qb_conj = conj.(qb)
        qz_conj = conj.(qz)
        px = p .* qx_conj
        py = p .* qy_conj
        pz = p .* qz_conj
        pb = (p .* qb_conj) ./ (N0.^2)
        return (px, py, pz, pb)
    end

    @inline function coeff_C(kx0,ky0,kz0, om0,s0, kx1,ky1,kz1,om1, s1, kx2,ky2,kz2, om2,s2, f0, N0)
        p0 = vec_p.(kx0, ky0, kz0, om0,s0, f0, N0)
        q1, q2 = vec_q.(kx1, ky1, kz1, om1,s1, f0, N0), vec_q.(kx2, ky2, kz2, om2,s2, f0, N0)
        px0, py0, pz0, pb0 = p0
        qx1, qy1, qz1, qb1 = q1
        qx2, qy2, qz2, qb2 = q2
        p0 = coeff_p.(kx0, ky0, kz0, om0,s0, f0, N0)
        p1 = coeff_p.(kx1, ky1, kz1, om1,s1, f0, N0)
        p2 = coeff_p.(kx2, ky2, kz2,om2,s2, f0, N0)
        q1k2 = qx1 .* kx2 .+ qy1 .* ky2 .+ qz1 .* kz2
        q2k1 = qx2 .* kx1 .+ qy2 .* ky1 .+ qz2 .* kz1
        px0_qx2 = px0 .* qx2
        py0_qy2 = py0 .* qy2
        pz0_qz2 = pz0 .* qz2
        pb0_qb2 = pb0 .* qb2
        px0_qx1 = px0 .* qx1
        py0_qy1 = py0 .* qy1
        pz0_qz1 = pz0 .* qz1
        pb0_qb1 = pb0 .* qb1

        coeff_C = 0.5 .* sqrt.(p0 .* p1 .* p2) ./ p0 .* ((px0_qx2 .+ py0_qy2 .+ pz0_qz2 .+ pb0_qb2) .* q1k2 .+ (px0_qx1 .+ py0_qy1 .+ pz0_qz1 .+ pb0_qb1) .* q2k1)
        return coeff_C
    end

    @inline function reso_triad_sum(kx0,ky0,kz0, om0,s0, E0, kx1,ky1,kz1, om1,s1, E1, kx2,ky2,kz2, om2,s2, E2, f0, N0, epsilon_h, epsilon_v)    
        C0 = coeff_C.(kx0,ky0,kz0, om0, s0,  kx1,ky1,kz1, om1, s1, kx2,ky2,kz2,om2, s2, f0, N0)
        Ep0 = E0.(kx0,ky0,kz0, s0, N0, f0, epsilon_h, epsilon_v)
        Ep1 = E1.(kx1,ky1,kz1, s1, N0, f0, epsilon_h, epsilon_v)
        Ep2 = E2.(kx2,ky2,kz2, s2, N0, f0, epsilon_h, epsilon_v)
        C3 = 4.0 .* (Ep1 .* Ep2 .- Ep0 .* Ep1 .* om2 ./ om0 .- Ep0 .* Ep2 .* om1 ./ om0) .* (abs.(C0)).^2
        gs = kz2 ./ om2 .* (f0.^2 .- om2.^2) ./ (kx2.^2 .+ ky2.^2 .+ kz2.^2) .- kz1 ./ om1 .* (f0.^2 .- om1.^2) ./ (kx1.^2 .+ ky1.^2 .+ kz1.^2)
        C3 = 2.0 .* π .* C3 ./ abs.(gs)
        return C3
    end

    @inline function reso_triad_diff(kx0,ky0,kz0, om0, s0, E0, kx1,ky1,kz1, om1, s1, E1, kx2,ky2,kz2, om2, s2, E2, f0, N0, epsilon_h, epsilon_v)
        C0 = coeff_C.(kx0,ky0,kz0,om0, s0, kx1,ky1,kz1, om1,s1, .-kx2,.-ky2,.-kz2,om2, .-s2, f0, N0)
        Ep0 = E0.(kx0,ky0,kz0, s0, N0, f0, epsilon_h, epsilon_v)
        Ep1 = E1.(kx1,ky1,kz1, s1, N0, f0, epsilon_h, epsilon_v)
        Ep2 = E2.(kx2,ky2,kz2, s2, N0, f0, epsilon_h, epsilon_v)
        C3 = 4.0 .* (Ep1 .* Ep2 .+ Ep0 .* Ep1 .* om2 ./ om0 .- Ep0 .* Ep2 .* om1 ./ om0) .* (abs.(C0)).^2
        gs = kz2 ./ om2 .* (f0.^2 .- om2.^2) ./ (kx2.^2 .+ ky2.^2 .+ kz2.^2) .- kz1 ./ om1 .* (f0.^2 .- om1.^2) ./ (kx1.^2 .+ ky1.^2 .+ kz1.^2)
        C3 = 2.0 .* π .* C3 ./ abs.(gs)
        return C3
    end

    @inline function Ep(kx,ky,kz, ss, N0, f0, epsilon_h,epsilon_z)
        kkx = abs.(kx) .+ epsilon_h
        kky = abs.(ky) .+ epsilon_h
        kkz = abs.(kz) .+ epsilon_z

        t = 2
        s = 2
        E0 = 3e-3
        nb = 2 ./ π
        m_star = 0.01
        na = s .* gamma(t ./ s) ./ gamma(1 ./ s) ./ gamma((t .- 1) ./ s)
        fac = E0 .* nb .* na .* f0 .* sqrt.(N0.^2 .- f0.^2)
        KK2 = (kkx.^2 .+ kky.^2)
        Ksqr = KK2 .+ kkz.^2
        om2 = (N0.^2 .* KK2 .+ f0.^2 .* kkz.^2) ./ Ksqr
        c2star = N0.^2 ./ m_star.^2
        ms = sqrt.((N0.^2 .- om2) ./ c2star)
        Atl = (1.0 .+ abs.(kkz ./ ms).^s).^(.-t ./ s)
        energy = fac .* Atl .* kkz.^2 ./ (4.0 .* π .* ms)
        GM = ifelse.((kx.^2 .+ ky.^2 .+ kz.^2) .== 0 , 0.0 , energy ./ ((N0.^2 .* KK2 .+ f0.^2 .* kkz.^2).*sqrt.(Ksqr)) ./ (sqrt.(KK2)))
        return GM
    end

    function om_kz_sum(kz, kx0, ky0, kz0, kx1, ky1, s, f0, N0)
        om0 = sqrt.(omega2_true.(kx0, ky0, kz0, s, f0, N0))
        om1 = sqrt.(omega2_true.(kx1, ky1, kz, s, f0, N0))
        kx2 = kx0 .- kx1
        ky2 = ky0 .- ky1
        kz2 = kz0 .- kz
        om2 = sqrt.(omega2_true.(kx2, ky2, kz2, s, f0, N0))
        om_kz_sum = om0 .- om1 .- om2
        return om_kz_sum
    end

    function om_kz_diff(kz, kx0, ky0, kz0, kx1, ky1, s, f0, N0)
        om0 = sqrt.(omega2_true.(kx0, ky0, kz0, s, f0, N0))
        om1 = sqrt.(omega2_true.(kx1, ky1, kz, s, f0, N0))
        kx2 = .-kx0 .+ kx1
        ky2 = .-ky0 .+ ky1
        kz2 = .-kz0 .+ kz
        om2 = sqrt.(omega2_true.(kx2, ky2, kz2, s, f0, N0))
        om_kz_diff = om0 .- om1 .+ om2
        return om_kz_diff
    end


    @inline function root_find_sp(kx0, ky0, kz0, kx1, ky1, s, f0, N0, z_min, z_max)
        root_sum_p(kx0, ky0, kz0, kx1, ky1) = fzero(kz -> om_kz_sum.(kz, kx0, ky0, kz0, kx1, ky1, s, f0, N0), (z_min,z_max))#(1e-24,1), xatol=1e-10, rtol=1e-10)
        kz1_b = root_sum_p.(kx0, ky0, kz0, kx1, ky1)
        return(kz1_b)
    end

    @inline function root_find_sm(kx0, ky0, kz0, kx1, ky1, s, f0, N0, z_min, z_max)
        root_sum_m(kx0, ky0, kz0, kx1, ky1) = fzero(kz -> om_kz_sum.(kz, kx0, ky0, kz0, kx1, ky1, s, f0, N0), (-z_max,-z_min))#(-1e-24,-1), xatol=1e-10, rtol=1e-10)
        kz1_b = root_sum_m.(kx0, ky0, kz0, kx1, ky1)
        return(kz1_b)
    end
    @inline function root_find_dp(kx0, ky0, kz0, kx1, ky1, s, f0, N0, z_min, z_max)
        root_diff_p(kx0, ky0, kz0, kx1, ky1) = fzero(kz -> om_kz_diff.(kz, kx0, ky0, kz0, kx1, ky1, s, f0, N0), (z_min,z_max)) #(1e-24,1), xatol=1e-10, rtol=1e-10)
        kz1_b = root_diff_p.(kx0, ky0, kz0, kx1, ky1)
        return(kz1_b)
    end
    @inline function root_find_dm(kx0, ky0, kz0, kx1, ky1, s, f0, N0, z_min, z_max)
        root_diff_m(kx0, ky0, kz0, kx1, ky1) = fzero(kz -> om_kz_diff.(kz, kx0, ky0, kz0, kx1, ky1, s, f0, N0), (-z_max,-z_min))#, xatol=1e-10, rtol=1e-10)
        kz1_b = root_diff_m.(kx0, ky0, kz0, kx1, ky1)
        return(kz1_b)
    end


    function scattering_loop(i,X0,Z0,X1,Y1,x,y,z,f0,N0,epsilon_h,epsilon_z, z_max)
        Y0=0.0

        # SUM INTERACTIONS
        # positive solution through bracketing
        my_cond = om_kz_sum.(0, X0,Y0,Z0, X1, Y1, 1, f0, N0) .* om_kz_sum.(z_max, X0,Y0,Z0, X1, Y1, 1, f0, N0) .>= 0
        kkx1 = X1[my_cond.==0]
        kky1 = Y1[my_cond.==0]
        Z1_0= root_find_sp(X0,Y0,Z0,kkx1,kky1, 1, f0, N0,1e-24,z_max)
        X2 = X0  .- kkx1
        Y2 = Y0 .- kky1
        Z2 = Z0  .- Z1_0

        # Zylinder condition
        n = ((kkx1.^2 .+ kky1.^2 .+ Z1_0.^2 .> 0.0) .&& 
            (X2.^2 .+ Y2.^2 .+ Z2.^2 .> 0.0) .&& 
            (X2.^2 .+ Y2.^2 .<= x[end].^2 ) .&& 
            (Z2.^2 .<= z[end].^2) )
        kx0,ky0,kz0 = (X0.*ones(Float64, size(kkx1)[1]))[n], (Y0.*ones(Float64, size(kkx1)[1]))[n], (Z0.*ones(Float64, size(kkx1)[1]))[n]
        kx1,ky1,kz1 = kkx1[n], kky1[n], Z1_0[n]
        kx2,ky2,kz2 = X2[n], Y2[n], Z2[n]

        if isempty(kx0)
            wave_sum_p_f = 0.0
            wave_sum_p_N = 0.0
            wave_sum_p_ = 0.0
        else
            n_f1 = (kx1.^2 .+ ky1.^2 .== 0)
            n_f2 = (kx2.^2 .+ ky2.^2 .== 0)
            n_N1 = (kz1.^2 .< 1e-12)
            n_N2 = (kz2.^2 .< 1e-12)
            n_f = (n_f1 .&& n_N2 .== 0) .|| (n_f2 .&& n_N1 .== 0)
            n_N = (n_N1 .&& n_f2 .== 0) .|| (n_N2 .&& n_f1 .== 0)
            om0 = sqrt.(omega2_true.(kx0, ky0, kz0, 1, f0, N0))
            om1 = sqrt.(omega2_true.(kx1, ky1, kz1, 1, f0, N0))
            om2 = sqrt.(omega2_true.(kx2, ky2, kz2, 1, f0, N0))
            wave_sum_p =  reso_triad_sum.(kx0,ky0,kz0, om0, +1, Ep, kx1,ky1,kz1,om1, +1, Ep, kx2,ky2,kz2,om2, +1, Ep, f0, N0, epsilon_h, epsilon_z)
            wave_sum_p_f = nansum(n_f .* wave_sum_p)
            wave_sum_p_N = nansum(n_N .* wave_sum_p)
            wave_sum_p_ = nansum(ifelse.(n_f .+ n_N .== 0, wave_sum_p, 0.0))   
        end
        
        # negatuve solution through bracketing
        my_cond = om_kz_sum.(-0.0, X0,Y0,Z0, X, Y, 1, f0, N0) .* om_kz_sum.(-z_max, X0,Y0,Z0, X, Y, 1, f0, N0) .>= 0
        kkx1 = X1[my_cond.==0]
        kky1 = Y1[my_cond.==0]
        Z1_1= root_find_sm(X0,Y0,Z0,kkx1,kky1, 1, f0, N0,1e-24,z_max)
        X2 = X0 .- kkx1
        Y2 = Y0 .- kky1
        Z2 = Z0 .- Z1_1

        # Zylinder condition
        n = ((kkx1.^2 .+ kky1.^2 .+ Z1_1.^2 .> 0.0) .&& 
        (X2.^2 .+ Y2.^2 .+ Z2.^2 .> 0.0) .&& 
        (X2.^2 .+ Y2.^2 .<= x[end].^2 ) .&& 
        (Z2.^2 .<= z[end].^2) )
        kx0,ky0,kz0 = (X0.*ones(Float64, size(kkx1)[1]))[n], (Y0.*ones(Float64, size(kkx1)[1]))[n], (Z0.*ones(Float64, size(kkx1)[1]))[n]
        kx1,ky1,kz1 = kkx1[n], kky1[n], Z1_1[n]
        kx2,ky2,kz2 = X2[n], Y2[n], Z2[n]

        if isempty(kx0)
            wave_sum_m_f = 0.0  
            wave_sum_m_N = 0.0
            wave_sum_m_ = 0.0

        else
            n_f1 = (kx1.^2 .+ ky1.^2 .== 0)
            n_f2 = (kx2.^2 .+ ky2.^2 .== 0)
            n_N1 = (kz1.^2 .< 1e-12)
            n_N2 = (kz2.^2 .< 1e-12)
            n_f = (n_f1 .&& n_N2 .== 0) .|| (n_f2 .&& n_N1 .== 0)
            n_N = (n_N1 .&& n_f2 .== 0) .|| (n_N2 .&& n_f1 .== 0)
            om0 = sqrt.(omega2_true.(kx0, ky0, kz0, 1, f0, N0))
            om1 = sqrt.(omega2_true.(kx1, ky1, kz1, 1, f0, N0))
            om2 = sqrt.(omega2_true.(kx2, ky2, kz2, 1, f0, N0))
            wave_sum_m =  reso_triad_sum.(kx0,ky0,kz0, om0, +1, Ep, kx1,ky1,kz1,om1, +1, Ep, kx2,ky2,kz2,om2, +1, Ep, f0, N0, epsilon_h, epsilon_z)
            wave_sum_m_f = nansum(n_f .* wave_sum_m)
            wave_sum_m_N = nansum(n_N .* wave_sum_m)
            wave_sum_m_rest = nansum(ifelse.(n_f .+ n_N .== 0, wave_sum_m, 0.0))
            
        end
        
        ## DIFFERENCE INTERACTION
        # positive solution through bracketing
        my_cond = om_kz_diff.(0.0, X0,Y0,Z0, X1, Y1, 1, f0, N0) .* om_kz_diff.(z_max, X0,Y0,Z0, X1, Y1, 1, f0, N0) .>= 0
        kkx1 = X1[my_cond.==0]
        kky1 = Y1[my_cond.==0]
        Z1_2 = root_find_dp(X0,Y0,Z0,kkx1,kky1, 1, f0, N0,1e-24,z_max)
        X2 = .-X0 .+ kkx1
        Y2 = .-Y0 .+ kky1
        Z2 = .-Z0 .+ Z1_2

        # Zylinder condition
        n = ((kkx1.^2 .+ kky1.^2 .+ Z1_2.^2 .> 0.0) .&& 
        (X2.^2 .+ Y2.^2 .+ Z2.^2 .> 0.0) .&& 
        (X2.^2 .+ Y2.^2 .<= x[end].^2 ) .&& 
        (Z2.^2 .<= z[end].^2) )
        kx0,ky0,kz0 = (X0.*ones(Float64, size(kkx1)[1]))[n], (Y0.*ones(Float64, size(kkx1)[1]))[n], (Z0.*ones(Float64, size(kkx1)[1]))[n]
        kx1,ky1,kz1 = kkx1[n], kky1[n], Z1_2[n]
        kx2,ky2,kz2 = X2[n], Y2[n], Z2[n]

        if isempty(kx0)
            wave_diff_p_f = 0.0
            wave_diff_p_N = 0.0
            wave_diff_p_ = 0.0   
        else
            # Perform calculation here
            # For example, let's calculate the sum of the array
            n_f1 = (kx1.^2 .+ ky1.^2 .== 0)
            n_f2 = (kx2.^2 .+ ky2.^2 .== 0)
            n_N1 = (kz1.^2 .< 1e-12)
            n_N2 = (kz2.^2 .< 1e-12)
            n_f = (n_f1 .&& n_N2 .== 0) .|| (n_f2 .&& n_N1 .== 0)
            n_N = (n_N1 .&& n_f2 .== 0) .|| (n_N2 .&& n_f1 .== 0)
            om0 = sqrt.(omega2_true.(kx0, ky0, kz0, 1, f0, N0))
            om1 = sqrt.(omega2_true.(kx1, ky1, kz1, 1, f0, N0))
            om2 = sqrt.(omega2_true.(kx2, ky2, kz2, 1, f0, N0))
            wave_diff_p =  reso_triad_diff.(kx0,ky0,kz0, om0, +1, Ep, kx1,ky1,kz1,om1, +1, Ep, kx2,ky2,kz2,om2, +1, Ep, f0, N0, epsilon_h, epsilon_z)
            wave_diff_p_f = nansum(n_f .* wave_diff_p)
            wave_diff_p_N = nansum(n_N .* wave_diff_p)
            wave_diff_p_rest = nansum(ifelse.(n_f .+ n_N .== 0, wave_diff_p, 0.0))

        end

        # negative solution through bracketing
        my_cond = om_kz_diff.(0.0, X0,Y0,Z0, X1, Y1, 1, f0, N0) .* om_kz_diff.(-z_max, X0,Y0,Z0, X1, Y1, 1, f0, N0) .>= 0
        kkx1 = X1[my_cond.==0]
        kky1 = Y1[my_cond.==0]
        Z1_3 = root_find_dm(X0,Y0,Z0,kkx1,kky1, 1, f0, N0,1e-24,z_max)
        X2 = .-X0 .+ kkx1
        Y2 = .-Y0 .+ kky1
        Z2 = .-Z0 .+ Z1_3

        # Zylinder condition
        n = ((kkx1.^2 .+ kky1.^2 .+ Z1_3.^2 .> 0.0) .&& 
        (X2.^2 .+ Y2.^2 .+ Z2.^2 .> 0.0) .&& 
        (X2.^2 .+ Y2.^2 .<= x[end].^2 ) .&& 
        (Z2.^2 .<= z[end].^2) )
        kx0,ky0,kz0 = (X0.*ones(Float64, size(kkx1)[1]))[n], (Y0.*ones(Float64, size(kkx1)[1]))[n], (Z0.*ones(Float64, size(kkx1)[1]))[n]
        kx1,ky1,kz1 = kkx1[n], kky1[n], Z1_3[n]
        kx2,ky2,kz2 = X2[n], Y2[n], Z2[n]

        if isempty(kx0)
            wave_diff_m_f = 0.0
            wave_diff_m_N = 0.0
            wave_diff_m_ = 0.0
        else
            # Perform calculation here
            # For example, let's calculate the sum of the array
            n_f1 = (kx1.^2 .+ ky1.^2 .== 0)
            n_f2 = (kx2.^2 .+ ky2.^2 .== 0)
            n_N1 = (kz1.^2 .< 1e-12)
            n_N2 = (kz2.^2 .< 1e-12)
            n_f = (n_f1 .&& n_N2 .== 0) .|| (n_f2 .&& n_N1 .== 0)
            n_N = (n_N1 .&& n_f2 .== 0) .|| (n_N2 .&& n_f1 .== 0)
            om0 = sqrt.(omega2_true.(kx0, ky0, kz0, 1, f0, N0))
            om1 = sqrt.(omega2_true.(kx1, ky1, kz1, 1, f0, N0))
            om2 = sqrt.(omega2_true.(kx2, ky2, kz2, 1, f0, N0))
            wave_diff_m =  reso_triad_diff.(kx0,ky0,kz0, om0, +1, Ep, kx1,ky1,kz1,om1, +1, Ep, kx2,ky2,kz2,om2, +1, Ep, f0, N0, epsilon_h, epsilon_z)

            wave_diff_m_f = nansum(n_f .* wave_diff_m)
            wave_diff_m_N = nansum(n_N .* wave_diff_m)
            wave_diff_m_ = nansum(ifelse.(n_f .+ n_N .== 0, wave_diff_m, 0.0))
        end
        return(X0[1],
                Z0[1],
                wave_sum_p_f.+wave_sum_m_f,
                wave_sum_p_N.+wave_sum_m_N,
                wave_sum_p_.+wave_sum_m_, 
                wave_diff_p_f.+wave_diff_m_f, 
                wave_diff_p_N.+wave_diff_m_N,
                wave_diff_p_.+wave_diff_m_)
    end
end

println("Communication completed")
println("Setting up domain")
flush(stdout)
nx=300
ny=nx
nz=450
Lx = 75e3
Ly = Lx
Lz = 6000.
f0 = 1e-4
N0 = 50*f0
highx = 2.0*pi/Lx * (nx-1) 
highy = highx
highz = 2.0*pi/Lz * (nz-1)

x_pos = range(0,highx,length = nx+1)
x_neg = reverse(.-x_pos[2:end])
x = vcat(x_neg,x_pos)
y = x
z_pos = range(0.0,highz,length = nz+1)
z_neg = reverse(.-z_pos[2:end])
z = vcat(z_neg,z_pos)
z_max = z[end]
dV = (x[2]-x[1])*(y[2]-y[1])
X,Y = zeros(Float64, size(x)[1],size(x_pos)[1]),  zeros(Float64, size(x)[1],size(x_pos)[1])#,  zeros(Float64, nx*ny)

for i in 1:size(x)[1]
    for j in 1:size(x_pos)[1]
    #for k in 1:nz
        X[i,j] = x[i]
        Y[i,j] = x_pos[j]
        
        #Z[i*j*k] = z[k]
    #end
    end
end
X,Y = vec(X), vec(Y)

x_2d = x_pos
z_2d = z_pos
Z2d,X2d = zeros(Float64, size(z_2d)[1],size(x_2d)[1]),  zeros(Float64, size(z_2d)[1],size(x_2d)[1])
for i in 1:size(z_2d)[1]
    for j in 1:size(x_2d)[1]
        X2d[i,j] = x_2d[j]
        Z2d[i,j] = z_2d[i]
    end
end
X2d,Z2d = vec(X2d), vec(Z2d)


N_len = Int(size(x_2d)[1]*size(z_2d)[1])
epsilon_h = x_pos[2]*0.5 
epsilon_z = z_pos[2]*0.5
println("Domain: nx=$nx, ny=$ny, nz=$nz")
println("Spectral domain #grids: $(nx*ny*nz)")
flush(stdout)

# Initialize an empty array to collect results
collected_results =  @showprogress @distributed (vcat) for i in 1:N_len #n_start:n_end#
    scattering_loop(i,X2d[i],Z2d[i],X,Y,x,y,z,f0,N0,epsilon_h,epsilon_z, z_max)    
end
# Separate the collected results into individual arrays
kx_local = [df[1] for df in collected_results]
kz_local = [df[2] for df in collected_results]
wave_sum_f0 = [(df[3].* dV ) for df in collected_results]
wave_sum_N0 = [(df[4].* dV ) for df in collected_results]
wave_sum_ =[(df[5].* dV ) for df in collected_results]
wave_diff_f0 = [(2.0 * df[6].* dV) for df in collected_results]
wave_diff_N0 = [(2.0 .* df[7].* dV ) for df in collected_results]
wave_diff_ =[(2.0 .* df[8].* dV ) for df in collected_results]


println("saving model")
flush(stdout)
name = "M3.npz"
npzwrite(name, Dict(
    "kx" => kx_local,
    "kz" => kz_local,
    "ww_sum_f0" => wave_sum_f0,
    "ww_sum_N0" => wave_sum_N0,
    "ww_sum" => wave_sum_,
    "ww_diff_f0" => wave_diff_f0,
    "ww_diff_N0" => wave_diff_N0,
    "ww_diff" => wave_diff_))
println("Model saved")
flush(stdout)
