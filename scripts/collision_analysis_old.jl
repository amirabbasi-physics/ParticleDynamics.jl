### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 2dc4841f-00b3-429c-88f1-303f7015fbb1
using PyPlot, DelimitedFiles, Statistics, Printf

# ╔═╡ 0218d62c-3f94-4b98-a1a4-e4fbe520fadc
begin
	cd(@__DIR__)
	rc("text",usetex = true)
	rc("font",family="Times New Roman")
	navy_blue = "#194184"
	# Function to process a single log file
	function process_file(filepath, dt)
	    lines_count = countlines(filepath)
	    if lines_count <= 20
	        return NaN, NaN, NaN, NaN, NaN  # Skip files without enough data
	    end
	    
	    data = readdlm(filepath, skipstart=21)
	    E_pot = mean(data[:, 3])
	    cold_cold = mean(data[:, end-2]) .* dt
	    hot_cold = mean(data[:, end-1]) .* dt
	    hot_hot = mean(data[:, end]) .* dt
	    return E_pot, cold_cold, hot_cold, hot_hot
	end
	
	# Main function to process directories and plot data
	function main()
	    directories = ["alpha_100.0", "alpha_1000.0", "alpha_10000.0"]
	    results = Dict()
		dt = 2.0e-6 * 1e6

		for dir in directories
    		alpha = parse(Float64, split(dir, "_")[2])
    		files = filter(f -> endswith(f, ".log"), readdir(dir))
    		for file in files
        		m = match(r"fraction-(\d+\.\d+)", file)
        		n = match(r"Npart,(\d+)", file) 
        		if m !== nothing && n !== nothing
            		phi = parse(Float32, m.captures[1])
            		num_particles = parse(Int, n.captures[1])  # Parse the number of particles as an integer
            		E_pot_avg, CC_avg, CH_avg, HH_avg = process_file(joinpath(dir, file), dt)
            		coll_avg =  (CC_avg .+ CH_avg .+ HH_avg) / (num_particles)
            		push!(results, (alpha, phi) => (E_pot_avg/ num_particles, CC_avg, CH_avg, HH_avg, coll_avg))
        		end
    		end
		end

		

		
	    # Adjusting plot parameters similar to matplotlib customization in Python
	    PyPlot.rc("font", size=12) # Example to set default font size
	    PyPlot.rc("axes", titlesize=14) # Title font size
	    PyPlot.rc("axes", labelsize=12) # Axes label font size
	    PyPlot.rc("xtick", labelsize=10) # X-tick label font size
	    PyPlot.rc("ytick", labelsize=10) # Y-tick label font size
	    PyPlot.rc("legend", fontsize=12) # Legend font size
	
	    alphas = sort(unique(first.(keys(results))))
	
	    # Creating subplots
	    fig, axs = subplots(2, 1, figsize=(12, 15))

		phi = 0.005:0.0001:0.85

		axs[1].plot(phi , 100sqrt(2π) * phi ./ (1.0 .- phi ) .^ 1.5, linewidth = 4.0, linestyle = "--", color= navy_blue, label = "Fit: "*L"\frac{c\,\phi}{(1-\phi)^{3/2}}")


		

		axs[1].plot(phi, 100sqrt(2π) * phi , linewidth = 4.0, linestyle = "--", color= "Red", label = "Dilute regime theory")


		
		axs[2].plot(phi, 1e-6*100sqrt(2π) * phi ./ (1.0 .- phi ) .^ 1.5   , linewidth = 4.0, linestyle = "--", color= navy_blue, label = "Fit:  "*L"\frac{c\,\phi}{(1-\phi)^{3/2}}")
		
		axs[2].plot(phi, 1e-6*100sqrt(2π) * phi , linewidth = 4.0, linestyle = "--", color= "Red", label = "Dilute regime theory")

		#####################################################
		# 													#
		# 				  Simulation Data 					#
		# 													#
		#####################################################
		
	    for (idx, alpha) in enumerate(alphas)
	        phi_data = [phi for (a, phi) in keys(results) if a == alpha]
	        E_pot_data = [results[(alpha, phi)][1] for phi in phi_data]
	        coll_avg_data = [results[(alpha, phi)][5] for phi in phi_data]
	
	        
	        axs[1].scatter(phi_data , coll_avg_data ./ sqrt(alpha), marker="o", s=250, label=L"$T_i/300$ = "*"$alpha")

			
			axs[2].scatter(phi_data, E_pot_data ./ (alpha^(1.5)), marker="o", s=250, label=L"$T_i/300$ = "*"$alpha")
	    end


		for ax in [axs[1], axs[2]]  # Assuming axs is your array of subplot axes
        # Set tick label font sizes
			for tick in ax.xaxis.get_major_ticks()
		        tick.tick1line.set_markersize(0)  # Hide major ticks if needed
		        tick.tick2line.set_markersize(0)
		        tick.label1.set_fontsize(30)  # Adjust fontsize for the label
		        tick.label2.set_fontsize(30)  # Adjust fontsize for the label
		    end
		
		    for tick in ax.yaxis.get_major_ticks()
		        tick.tick1line.set_markersize(0)  # Hide major ticks if needed
		        tick.tick2line.set_markersize(0)
		        tick.label1.set_fontsize(30)  # Adjust fontsize for the label
		        tick.label2.set_fontsize(30)  # Adjust fontsize for the label
		    end

        

        # Major and minor tick parameters
        ax[:tick_params](which="major", axis="x", direction="in", length=10, width=2.5)
        ax[:tick_params](which="minor", axis="x", direction="in", length=6, width=2.)
        ax[:tick_params](which="major", axis="y", direction="in", length=10, width=2.5)
        ax[:tick_params](which="minor", axis="y", direction="in", length=6, width=2)
    end

		for ax in [axs[1], axs[2]]
			ax.spines["left"].set_linewidth(3)
			ax.spines["right"].set_linewidth(3)
			ax.spines["top"].set_linewidth(3)
			ax.spines["bottom"].set_linewidth(3)
		end

		axs[1].set_xlabel(L"$\phi_t$", fontsize = 40)
	    axs[1].set_ylabel(L"$\kappa_{ii}\,\delta t/\sqrt{T_i}$", fontsize = 40)
	    #axs[2].set_title("Average Sum of Last Three Columns vs Phi")
	    axs[1].legend(fontsize = 30)
	    axs[1].grid(true)
	    axs[1].set_xscale("linear")
	    axs[1].set_yscale("log")
		
	    axs[2].set_xlabel(L"$\phi_t$", fontsize = 40)
	    axs[2].set_ylabel(L"$\langle E_{pot} \rangle/T_i^{3/2}$", fontsize = 40)
	    #axs[1].set_title("Average E_pot vs Phi")
	    axs[2].legend(fontsize = 30)
		axs[2].set_xscale("linear")
	    axs[2].set_yscale("log")
		axs[2].grid(true)
	

	
	    tight_layout()
		savefig("Collision_rate_2D_Maiti.pdf")
	    gcf()
	end
end

# ╔═╡ 57e7cc3f-ebbb-4393-85fa-4337a1e7e1ab
main()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PyPlot = "~2.11.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "f7b46d6d484f1564a92f1b50765328f51cf3ba10"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "0371ca706e3f295481cbf94c8c36692b072285c2"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.5"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"
"""

# ╔═╡ Cell order:
# ╟─2dc4841f-00b3-429c-88f1-303f7015fbb1
# ╠═0218d62c-3f94-4b98-a1a4-e4fbe520fadc
# ╠═57e7cc3f-ebbb-4393-85fa-4337a1e7e1ab
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
