using CSV
using DataFrames
using Gadfly
using MLJ
using MLJModels
using DecisionTree
using MLJDecisionTreeInterface

# Import the Iris dataset into a dataframe
iris = CSV.read("iris.csv", DataFrame)
println("Dataset loaded successfully!")

# Print separate columns
println(iris[!, :Species])

# Plotting

p = plot(iris, x=:SepalLength, y=:SepalWidth, color=:Species, 
        Geom.point, Geom.smooth, Guide.xlabel("Sepal Length"),
        Guide.ylabel("Sepal Width"), Guide.title("Iris Sepal Dimensions"))


p_box = plot(iris, x=:Species, y=:PetalLength, color=:Species,
             Geom.boxplot, Guide.xlabel("Species"),
             Guide.ylabel("Petal Length"), Guide.title("Iris Petal Length by Species"))

# Features Dataframe
X = select(iris, Not(:Species))
y = iris.Species
y = coerce(y, Multiclass)

# Split data into training and testing sets
train, test = MLJ.partition(eachindex(iris[!,:Species]), 0.8, shuffle=true, rng=5464)
X_train = iris[train, Not(:Species)]
y_train = coerce(iris[train, :Species], Multiclass)
X_test = iris[test, Not(:Species)]
y_test = coerce(iris[test, :Species], Multiclass)

model = @load DecisionTreeClassifier pkg=DecisionTree
forest = model()

rf = machine(forest, X_train, y_train)
MLJ.fit!(rf)

y_hat = MLJ.predict(rf, X_test)
y_pred = MLJ.mode.(y_hat)

accuracy = MLJ.accuracy(y_pred, y_test) 
println("Model Accuracy: ", round(accuracy*100, digits=2), "%")

cm = StatisticalMeasures.ConfusionMatrices.confmat(y_pred, y_test)


