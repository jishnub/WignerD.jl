using Documenter
using WignerD

DocMeta.setdocmeta!(WignerD, :DocTestSetup, :(using WignerD); recursive=true)

makedocs(;
    modules=[WignerD],
    authors="Jishnu Bhattacharya",
    repo="https://github.com/jishnub/WignerD.jl/blob/{commit}{path}#L{line}",
    sitename="WignerD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jishnub.github.io/WignerD.jl",
        assets=String[],
    ),
    pages=[
        "WignerD" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jishnub/WignerD.jl",
)
