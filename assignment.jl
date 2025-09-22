using Pkg
using CSV
using DataFrames
using Statistics
using MLJ
using BetaML
using StatisticalMeasures.ConfusionMatrices

# Load the dataset
println("Loading bank loan dataset...")
df = CSV.read("bank-loan-dataset.csv", DataFrame)
println("Dataset loaded successfully!")
println("Dataset shape: ", size(df))
println("First 5 rows:")
println(first(df, 5))

# Prepare features and target
println("\nPreparing data...")
x = select(df, Not(:default))  # All columns except 'default'
y = df.default  # Target variable

# Convert target to proper format for MLJ
y = coerce(y, Multiclass)

println("Features: ", names(x))
println("Target distribution:")
println("True: ", count(==(true), y), ", False: ", count(==(false), y))

# Split data into training and testing sets (80/20 split)
println("\nSplitting data...")
train_indices, test_indices = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=42)

x_train = x[train_indices, :]
y_train = y[train_indices]
x_test = x[test_indices, :]
y_test = y[test_indices]

println("Training set size: ", size(x_train))
println("Test set size: ", size(x_test))

# Load and train RandomForestClassifier from BetaML
println("\nTraining RandomForestClassifier...")
model = @load RandomForestClassifier pkg=BetaML
rf_model = model()

# Create machine and fit
machine_rf = machine(rf_model, x_train, y_train)
MLJ.fit!(machine_rf)

# Make predictions
println("\nMaking predictions...")
y_pred_prob = MLJ.predict(machine_rf, x_test)
y_pred = MLJ.mode.(y_pred_prob)

# Calculate evaluation metrics
println("\nEvaluating model performance...")

# Confusion Matrix
cm = confusion_matrix(y_pred, y_test)
println("Confusion Matrix:")
println(cm)

# Calculate metrics
accuracy = MLJ.accuracy(y_pred, y_test)
precision_score = MLJ.multiclass_precision(y_pred, y_test) |> mean
recall_score = MLJ.multiclass_recall(y_pred, y_test) |> mean
f1 = MLJ.multiclass_f1score(y_pred, y_test) |> mean

println("\nModel Performance Metrics:")
println("Accuracy: ", round(accuracy, digits=4))
println("Precision: ", round(precision_score, digits=4))
println("Recall: ", round(recall_score, digits=4))
println("F1-Score: ", round(f1, digits=4))

# Explanation of metrics and quality assessment
println("\nMetric Explanations:")
println("- Accuracy: Overall correctness - $(round(accuracy*100, digits=2))% of predictions were correct")
println("- Precision: Of predicted defaults, $(round(precision_score*100, digits=2))% were actually defaults")
println("- Recall: Of actual defaults, $(round(recall_score*100, digits=2))% were correctly identified")
println("- F1-Score: Balanced measure combining precision and recall")

println("\nQuality Assessment:")
if accuracy > 0.8 && f1 > 0.7
    println("The model shows good performance with high accuracy and balanced precision/recall.")
    println("This indicates reliable loan default prediction capability.")
elseif accuracy > 0.7
    println("The model shows moderate performance. Accuracy is acceptable but could be improved.")
else
    println("The model performance needs improvement before deployment.")
end