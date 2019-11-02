__precompile__()
module COSMO

using SparseArrays, LinearAlgebra, SuiteSparse, QDLDL, Pkg


export assemble!, warmStart!, empty_model!

const DefaultFloat = Float64
const DefaultInt   = LinearAlgebra.BlasInt


include("./kktsolver.jl")
# optional dependencies
if in("Pardiso",keys(Pkg.installed()))
    include("./kktsolver_pardiso.jl")
end
include("./kktsolver_bo.jl")

include("./algebra.jl")
include("./projections.jl")
include("./settings.jl")            # TODO: unmodified - revisit
include("./trees.jl")
include("./types.jl")               # some types still need tidying
include("./constraint.jl")          # TODO: unmodified - revisit
include("./parameters.jl")          # TODO: unmodified - revisit
include("./residuals.jl")           # TODO: unmodified - revisit
include("./scaling.jl")             # TODO: set scaling / E scaling is broken
include("./infeasibility.jl")       # TODO: stylistic fixes needed
include("./chordal_decomposition.jl")
include("./printing.jl")            # TODO: unmodified - revisit
include("./setup.jl")               # TODO: unmodified - revisit (short - consolidate?)
include("./solver.jl")              # TODO: unmodified - revisit
include("./interface.jl")           # TODO: unmodified - revisit
include("./MOIWrapper.jl")


end #end module
