using LinearAlgebra
using Distributions
using DelimitedFiles



function findClosestCentroid(centroids, p)
    idxClosestCentr = -1
    minDist = typemax(Float32)

    for (i, x_i) in zip(1:length(centroids), centroids)
        dist = norm(x_i - p)
        if dist < minDist
            idxClosestCentr = i
            minDist = dist
        end
    end
    return idxClosestCentr, minDist
end


function fixedPointIter(centroids, xs)
    N  = length(centroids)
    M = length(xs)

    localMean = zeros(N, 2)
    localCount = zeros(N)
    localDist = 0.

    for x in xs
        idx, dist = findClosestCentroid(centroids, x)
        localMean[idx] = localMean[idx] .+ x
        localDist += dist^2
        localCount[idx] += 1
    end

    for i in 1:N
        if localCount[i] > 0
            centroids[i] = localMean[i] / Float32(localCount[i])
        end
    end

    probas = localCount / Float32(M)
    distortion = localDist / Float32(2. * M)
    return centroids, probas, distortion
end




function Lloyd(N, M, nIter)
    centroids = rand(Normal(0., 1.), (N, 2))
    # probas = zeros(nIter, M)
    # distortion = zeros(nIter)

    open("res.txt", "w") do io
        for t in 1:nIter
            x = rand(Normal(0., 1.), (M, 2))
            centroids, probas, distortion = fixedPointIter(centroids, x)
            writedlm(io, [centroids, probas, distortion])
        end
    end

    return centroids
end

N = 50
M = 5000
nIter = 100


Lloyd(N, M, nIter)

function readResFile(file_name)
    file = readdlm(file_name)
    r, c = size(file)
    N = Int(c / 2)
    nIter = Int(r / 3)

    centroids = zeros(nIter, N , 2)
    probs = zeros(nIter, c)
    distor = zeros(nIter)

    for i in 0:(nIter-1)
        centroids[i + 1, :, :] = reshape(file[3 * i + 1, :], (N, 2))
        probs[i + 1, :] = file[3 * i + 2, :]
        distor[i + 1] = file[3 * i + 3, 1]
    end

    return centroids, probs, distor
end

file_name = "res.txt"

centroids, probas, distor = readResFile(file_name)


using Plots, LaTeXStrings
using GeometricalPredicates: Point 

p = plot(1:100, distor, color = "crimson", lw = 2, ylabel = L"d_N^X", label = "Distortion")
savefig(p, "plot.pdf")

scatter(centroids[end, :, :][:, 1], centroids[end, :, :][:, 2], color = "crimson", markerstrokecolor = "black", markerstrokewidth = 2)


using VoronoiCells
using GeometryBasics
using Distributions

rect = Rectangle(Point(-5., -5.), Point(5., 5.))
points = Point.(centroids[end, :, :][:, 1],centroids[end, :, :][:, 2])
tess = voronoicells(points, rect)


using ColorSchemes

plot(tess, color = "black", lw = 2, primary = false)
scatter!(getx.(points), gety.(points), color = "darkred", label = "")
sum(probas[1, :])




plot(tess, primary = false, palette = reverse(palette(:viridis)), fill_z = reshape(probas[end, :], (1, 100)))
scatter!(getx.(points), gety.(points), color = "darkred", label = "")

probas[end, :]
size(tess.Cells)




# % Plots 
anim = @animate for i in 1:size(centroids, 1)
    #Random.seed!(123)
    points = Point.(centroids[i, :, :][:, 1],centroids[i, :, :][:, 2])
    tess = voronoicells(points, rect)
    plot(tess, fill = (true, :viridis), lw = 2, primary = false)
    scatter!(getx.(points), gety.(points), color = "darkred", label = "")

end
gif(anim, "tessell.gif")

π

# % Kohonen :

function lr(N, n)
    a = 4. * N
    b = π^2 / float(N * N)
    return a / float(a + b * (n + 1.))
end


function applyMGradientDescentSteps(centroids, xs, count, distortion, init_n)
    N = size(centroids, 1)

    for  x in  xs
        gamma_n = lr(N, init_n + n)

        # % Competition
        idx, l2Dist = findClosestCentroid(centroids, x)

        # % Learning
        centroids[idx] = centroids[idx] - gamma_n * (centroids[idx] - x)

        # % update distortion
        distortion = (1 - gamma_n) * distortion + .5 * gamma_n * l2Dist^2

        # % update cnter for probas computations
        count[idx] = count[idx] + 1
    end

    return centroids, count, distortion 
end



function Kohonen(N, n, nIter)

    M = Int(n / nIter)

    # % Init
    centroids = rand(Normal(0., 1), (N, 2))
    count = zeros(nIter)
    distortion = 0.

    open("kohonen.txt", "w") do io
        for t in 1:nIter
            println("Iter : ", t)
            xs = rand(Normal(0., 1.), (M, 2))

            centroids, count, distortion = applyMGradientDescentSteps(centroids, xs, count, distortion, t * M)
            
            writedlm(io, [centroids, distortion])
        end
    end

    probas = count / sum(count)
    return centroids, probas, distortion
end

N = 50
nIter = 100
M = 5000

c, p, d = Kohonen(N, M * nIter, nIter)

