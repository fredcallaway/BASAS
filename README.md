# BASAS

This is starter code for simulating the model from [this paper](https://direct.mit.edu/opmi/article/doi/10.1162/opmi.a.3/131590/Considering-What-We-Know-and-What-We-Don-t-Know).


## Installation

Install [juliaup](https://github.com/JuliaLang/juliaup). If you're on Max/Linux, you can run this command:

    curl -fsSL https://install.julialang.org | sh

Install packages:

    julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"


## Usage

Run the example script:

    julia example.jl

Note that julia is slow to start up, so it's best to run the script from the REPL.

    julia
    
    julia> include("example.jl")

I recommend using the [julia-vscode](https://www.julia-vscode.org/) extension for vscode. It provides an interface much like RStudio.