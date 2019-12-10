for (T1,T2) in Base.Iterators.product((Int64,Float64),(Int64,Float64))
	precompile(X,(T2,T2))
end

precompile(coeffi,(Int64,))
precompile(coeffi,(Float64,))

precompile(djmatrix,(Int64,Float64))

precompile(Ylmatrix,(Int64,Tuple{Float64,Float64}))
precompile(Ylmatrix,(Int64,Tuple{Int64,Float64}))
precompile(Ylmatrix,(Int64,Tuple{Float64,Int64}))
precompile(Ylmatrix,(Int64,Tuple{Int64,Int64}))
