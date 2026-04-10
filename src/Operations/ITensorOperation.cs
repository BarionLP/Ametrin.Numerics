namespace Ametrin.Numerics.Operations;

public interface IUnaryTensorOperator<TState, TTensor>
    where TState : allows ref struct
    where TTensor : struct, ITensorLike<TTensor>
{
    public abstract static void ForwardTo(TState state, TTensor input, Dynamic<TTensor> output);
    public abstract static void BackwardTo(TState state, TTensor input, TTensor output, TTensor outputGradient, TTensor inputGradient);
}

public interface IBinaryTensorOperator<TState, TLeftTensor, TRightTensor, TOutputTensor>
    where TState : allows ref struct
    where TLeftTensor : struct, ITensorLike<TLeftTensor>
    where TRightTensor : struct, ITensorLike<TRightTensor>
    where TOutputTensor : struct, ITensorLike<TOutputTensor>
{
    public abstract static void ForwardTo(TState state, TLeftTensor left, TRightTensor right, Dynamic<TOutputTensor> output);
    public abstract static void BackwardTo(TState state, TLeftTensor left, TRightTensor right, TOutputTensor output, TOutputTensor outputGradient, ref TLeftTensor leftGradient, ref TRightTensor rightGradient);
}

public readonly ref struct AddTensorsOperator<TTensor> : IBinaryTensorOperator<Empty, TTensor, TTensor, TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    public static void ForwardTo(Empty _, TTensor left, TTensor right, Dynamic<TTensor> output)
    {
        NumericsDebug.AssertSameDimensions(left, right);
        output.SetSize(left);
        TensorPrimitives.Add(left.AsSpan(), right.AsSpan(), output.AsSpan());
    }

    public static void BackwardTo(Empty _, TTensor left, TTensor right, TTensor output, TTensor outputGradient, ref TTensor leftGradient, ref TTensor rightGradient)
    {
        leftGradient = outputGradient;
        rightGradient = outputGradient;
    }

    public static void Test()
    {
        // Span<Weight> input = default;
        // Span<Weight> softmaxed = default;
        // Span<Weight> relued = default;
        // Span<Weight> output = default;

        // SoftMaxOperation.ForwardTo(default, input, softmaxed);
        // LeakyReLUOperation.ForwardTo(0.1f, input, relued);
        // AddTensorsOperator.ForwardTo(default, softmaxed, relued, output);

        // Span<Weight> gradient = default;
        // Span<Weight> softmaxedGradient = default;
        // Span<Weight> reluedGradient = default;
        // Span<Weight> inputGradient = default;
        // Span<Weight> inputGradientB = default;

        // AddTensorsOperator.BackwardTo(default, softmaxed, relued, output, gradient, softmaxedGradient, reluedGradient);
        // LeakyReLUOperation.BackwardTo(0.1f, input, relued, reluedGradient, inputGradientB);
        // SoftMaxOperation.BackwardTo(default, input, softmaxed, softmaxedGradient, inputGradient);

        // AddTensorsOperator.ForwardTo(default, inputGradient, inputGradientB, inputGradient);

    }

    public static void Test2()
    {
        var input = new InputNode<Vector>();
        var softmaxed = new UnaryOperationNode<SoftMaxOperation, Empty, Vector> { Source = input, State = default };
        var relued = new UnaryOperationNode<LeakyReLUOperation, float, Vector> { Source = input, State = 0.1f };
        var output = new BinaryOperationNode<AddTensorsOperator<Vector>, Empty, Vector, Vector, Vector> { LeftSource = softmaxed, RightSource = relued, State = default, };

        output.Backward(output.Evaluate());
    }
}

public interface IOperationNode<TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    public TTensor Evaluate();
    public void Backward(TTensor outputGradient);
}

public sealed class InputNode<TTensor> : IOperationNode<TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    private TTensor input = TTensor.Empty;

    public void Set(TTensor input)
    {
        this.input = input;
    }

    public TTensor Evaluate()
    {
        return input;
    }

    public void Backward(TTensor outputGradient) { }
}

public sealed class LearnableWeightsNode<TTensor>(TTensor weights) : IOperationNode<TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    private readonly TTensor weights = weights;
    public readonly TTensor AccumulatedGradients = TTensor.OfSize(weights);

    public TTensor Evaluate() => weights;
    public void Backward(TTensor outputGradient)
    {
        TensorPrimitives.Add(AccumulatedGradients.AsSpan(), outputGradient.AsSpan(), AccumulatedGradients.AsSpan());
    }
}

public sealed class UnaryOperationNode<TOperator, TState, TTensor> : IOperationNode<TTensor>
    where TOperator : IUnaryTensorOperator<TState, TTensor>, allows ref struct
    where TTensor : struct, ITensorLike<TTensor>
{

    public required IOperationNode<TTensor> Source { get; init; }
    public required TState State { get; init; }

    private TTensor input;
    private readonly Dynamic<TTensor> output = new();
    private readonly Dynamic<TTensor> inputGradient = new();

    public TTensor Evaluate()
    {
        input = Source.Evaluate();
        TOperator.ForwardTo(State, input, output);
        return output;
    }

    public void Backward(TTensor outputGradient)
    {
        inputGradient.SetSize(input);

        TOperator.BackwardTo(State, input, output, outputGradient, inputGradient);

        Source.Backward(inputGradient);
    }
}

public sealed class BinaryOperationNode<TOperator, TState, TLeftTensor, TRightTensor, TOutputTensor> : IOperationNode<TOutputTensor>
    where TOperator : IBinaryTensorOperator<TState, TLeftTensor, TRightTensor, TOutputTensor>, allows ref struct
    where TLeftTensor : struct, ITensorLike<TLeftTensor>
    where TRightTensor : struct, ITensorLike<TRightTensor>
    where TOutputTensor : struct, ITensorLike<TOutputTensor>
{
    public required IOperationNode<TLeftTensor> LeftSource { get; init; }
    public required IOperationNode<TRightTensor> RightSource { get; init; }
    public required TState State { get; init; }

    private TLeftTensor inputLeft;
    private TRightTensor inputRight;
    private readonly Dynamic<TOutputTensor> output = new();
    private readonly Dynamic<TLeftTensor> leftGradientHandle = new();
    private readonly Dynamic<TRightTensor> rightGradientHandle = new();

    public TOutputTensor Evaluate()
    {
        inputLeft = LeftSource.Evaluate();
        inputRight = RightSource.Evaluate();
        TOperator.ForwardTo(State, inputLeft, inputRight, output);
        return output;
    }

    public void Backward(TOutputTensor outputGradient)
    {
        leftGradientHandle.SetSize(inputLeft);
        rightGradientHandle.SetSize(inputRight);

        var leftGradient = leftGradientHandle.Tensor;
        var rightGradient = rightGradientHandle.Tensor;

        TOperator.BackwardTo(State, inputLeft, inputRight, output, outputGradient, ref leftGradient, ref rightGradient);

        LeftSource.Backward(leftGradient);
        RightSource.Backward(rightGradient);
    }
}

public readonly ref struct SoftMaxOperation : IUnaryTensorOperator<Empty, Vector>
{
    public static int GetOutputSize(Vector input) => input.Count;
    public static void ForwardTo(Empty _, Vector input, Dynamic<Vector> outputStorage)
    {
        outputStorage.SetSize(input);
        input.SoftMaxTo(outputStorage);
    }

    public static void BackwardTo(Empty _, Vector input, Vector output, Vector outputGradient, Vector inputGradient)
    {
        var dot = TensorPrimitives.Dot(output.AsSpan(), outputGradient.AsSpan());
        TensorPrimitives.AddMultiply(outputGradient.AsSpan(), -dot, output.AsSpan(), inputGradient.AsSpan());
        // TensorPrimitives.Subtract(outputGradient, dot, inputGradient);
        // TensorPrimitives.Multiply(inputGradient, output, inputGradient);
    }
}

public readonly ref struct LeakyReLUOperation : IUnaryTensorOperator<Weight, Vector>
{
    public static int GetOutputSize(Vector input) => input.Count;
    public static void ForwardTo(Weight alpha, Vector input, Dynamic<Vector> outputStorage)
    {
        outputStorage.SetSize(input);
        SpanOperations.MapTo<LeakyReLUOp, Weight>(alpha, input.AsSpan(), outputStorage.AsSpan());
    }

    public static void BackwardTo(Weight alpha, Vector input, Vector output, Vector outputGradient, Vector inputGradient)
    {
        SpanOperations.MapTo<LeakyReLUGradientOp, Weight>(alpha, input.AsSpan(), outputGradient.AsSpan(), inputGradient.AsSpan());
    }

    private readonly ref struct LeakyReLUOp : IUnaryOperator<Weight>
    {
        public static Weight Invoke(in Weight alpha, Weight input) => input > 0 ? input : alpha * input;
        public static SimdVector Invoke(in Weight alpha, SimdVector input)
            => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), input, input * alpha);
    }

    private readonly ref struct LeakyReLUGradientOp : IBinaryOperator<Weight>
    {
        public static Weight Invoke(in Weight alpha, Weight input, Weight gradient) => input > 0 ? gradient : gradient * alpha;
        public static SimdVector Invoke(in Weight alpha, SimdVector input, SimdVector gradient)
            => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), gradient, gradient * alpha);
    }
}

public readonly ref struct MatrixVectorMultiplyOperation : IBinaryTensorOperator<Empty, Matrix, Vector, Vector>
{
    public static void ForwardTo(Empty _, Matrix left, Vector right, Dynamic<Vector> output)
    {
        output.SetSize(left.ColumnCount);
        left.MultiplyTo(right, output);
    }

    public static void BackwardTo(Empty _, Matrix left, Vector right, Vector output, Vector outputGradient, ref Matrix leftGradient, ref Vector rightGradient)
    {
        VectorHelper.MultiplyToMatrixTo(outputGradient, right, leftGradient);
        left.MultiplyTransposedTo(outputGradient, rightGradient);
    }

}

public readonly struct Empty;
