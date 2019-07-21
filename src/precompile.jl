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

for (T1,T2) in Base.Iterators.product((Int64,Float64),(Int64,Float64))
	for (T3,T4) in Base.Iterators.product((Int64,Float64),(Int64,Float64))
		type_n1 = Tuple{T1,T2}
		type_n2 = Tuple{T3,T4} 
		precompile(BiPoSH_s0,(Int64,Int64,Int64,Int64,Int64,type_n1,type_n2))
		precompile(BiPoSH_s0,(Int64,Int64,Int64,type_n1,type_n2))
		precompile(BiPoSH_s0,(Int64,Int64,UnitRange{Int64},Int64,Int64,type_n1,type_n2))
		precompile(BiPoSH_s0,(Int64,Int64,UnitRange{Int64},type_n1,type_n2))
	end
end
