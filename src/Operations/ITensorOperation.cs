using System.Buffers;

namespace Ametrin.Numerics.Operations;

public interface IUnaryTensorOperator<TState, TTensor>
    where TState : allows ref struct
    where TTensor : struct, ITensorLike<TTensor>, allows ref struct
{
    public abstract static int GetOutputSize(TTensor input);
    public abstract static TTensor ForwardTo(TState state, TTensor input, ArrayHandle outputHandle);
    public abstract static void BackwardTo(TState state, TTensor input, TTensor output, TTensor outputGradient, TTensor inputGradient);
}

public interface IBinaryTensorOperator<TState, TLeftTensor, TRightTensor, TOutputTensor>
    where TState : allows ref struct
    where TLeftTensor : allows ref struct
    where TRightTensor : allows ref struct
    where TOutputTensor : allows ref struct
{
    public abstract static int GetOutputSize(TLeftTensor left, TRightTensor right);
    public abstract static TOutputTensor ForwardTo(TState state, TLeftTensor left, TRightTensor right, ArrayHandle outputHandle);
    public abstract static void BackwardTo(TState state, TLeftTensor left, TRightTensor right, TOutputTensor output, TOutputTensor outputGradient, TLeftTensor leftGradient, TRightTensor rightGradient);
}

public readonly ref struct AddTensorsOperator<TTensor> : IBinaryTensorOperator<Empty, TTensor, TTensor, TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    public static int GetOutputSize(TTensor left, TTensor right) => left.FlatCount;

    public static TTensor ForwardTo(Empty _, TTensor left, TTensor right, ArrayHandle outputHandle)
    {
        NumericsDebug.AssertSameDimensions(left, right);
        var output = TTensor.OfSize(left, outputHandle);
        TensorPrimitives.Add(left.AsSpan(), right.AsSpan(), output.AsSpan());
        return output;
    }

    public static void BackwardTo(Empty _, TTensor left, TTensor right, TTensor output, TTensor outputGradient, TTensor leftGradient, TTensor rightGradient)
    {
        outputGradient.AsSpan().CopyTo(leftGradient.AsSpan());
        outputGradient.AsSpan().CopyTo(rightGradient.AsSpan());
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
        var input = new WeightsSource<Vector> { Tensor = Vector.Empty };
        var softmaxed = new UnaryOperation<SoftMaxOperation, Empty, Vector> { Source = input, State = default };
        var relued = new UnaryOperation<LeakyReLUOperation, float, Vector> { Source = input, State = 0.1f };
        var output = new BinaryOperation<AddTensorsOperator<Vector>, Empty, Vector, Vector, Vector> { LeftSource = softmaxed, RightSource = relued, State = default, };

        output.Backward(output.Evaluate());
    }
}

public interface IOperationNode<TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    public TTensor Evaluate();
    public void Backward(TTensor outputGradient);
}

public readonly struct WeightsSource<TTensor> : IOperationNode<TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    public required TTensor Tensor
    {
        get;
        init
        {
            field = value;
            AccumulatedGradients = TTensor.OfSize(value);
        }
    }
    public TTensor AccumulatedGradients { get; private init; }

    public TTensor Evaluate() => Tensor;
    public void Backward(TTensor outputGradient)
    {
        TensorPrimitives.Add(AccumulatedGradients.AsSpan(), outputGradient.AsSpan(), AccumulatedGradients.AsSpan());
    }
}

public sealed class UnaryOperation<TOperator, TState, TTensor> : IOperationNode<TTensor>
    where TOperator : IUnaryTensorOperator<TState, TTensor>, allows ref struct
    where TTensor : struct, ITensorLike<TTensor>
{

    public required IOperationNode<TTensor> Source { get; init; }
    public required TState State { get; init; }

    private TTensor input;
    private readonly DynamicArrayHandle outputHandle = new();
    private TTensor output;

    public TTensor Evaluate()
    {
        input = Source.Evaluate();
        var outputSize = TOperator.GetOutputSize(input);
        outputHandle.SetCount(outputSize);
        output = TOperator.ForwardTo(State, input, outputHandle.Handle);
        return output;
    }

    public void Backward(TTensor outputGradient)
    {
        using var inputGradientHandle = ArrayPool<Weight>.Shared.RentNumerics(input.FlatCount);
        var inputGradient = TTensor.OfSize(input, inputGradientHandle);

        TOperator.BackwardTo(State, input, output, outputGradient, inputGradient);

        Source.Backward(inputGradient);
    }
}

public sealed class BinaryOperation<TOperator, TState, TLeftTensor, TRightTensor, TOutputTensor> : IOperationNode<TOutputTensor>
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
    private readonly DynamicArrayHandle outputHandle = new();
    private TOutputTensor output;

    public TOutputTensor Evaluate()
    {
        inputLeft = LeftSource.Evaluate();
        inputRight = RightSource.Evaluate();
        var outputSize = TOperator.GetOutputSize(inputLeft, inputRight);
        outputHandle.SetCount(outputSize);
        return output = TOperator.ForwardTo(State, inputLeft, inputRight, outputHandle.Handle);
    }

    public void Backward(TOutputTensor outputGradient)
    {
        using var inputLeftGradientHandle = ArrayPool<Weight>.Shared.RentNumerics(inputLeft.FlatCount);
        var inputLeftGradient = TLeftTensor.OfSize(inputLeft, inputLeftGradientHandle);

        using var inputRightGradientHandle = ArrayPool<Weight>.Shared.RentNumerics(inputRight.FlatCount);
        var inputRightGradient = TRightTensor.OfSize(inputRight, inputRightGradientHandle);

        TOperator.BackwardTo(State, inputLeft, inputRight, output, outputGradient, inputLeftGradient, inputRightGradient);

        LeftSource.Backward(inputLeftGradient);
        RightSource.Backward(inputRightGradient);
    }
}

public readonly ref struct SoftMaxOperation : IUnaryTensorOperator<Empty, Vector>
{
    public static int GetOutputSize(Vector input) => input.Count;
    public static Vector ForwardTo(Empty _, Vector input, ArrayHandle outputStorage)
    {
        var output = Vector.OfSize(input, outputStorage);
        input.SoftMaxTo(output);
        return output;
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
    public static Vector ForwardTo(Weight alpha, Vector input, ArrayHandle outputStorage)
    {
        var output = Vector.OfSize(input, outputStorage);
        SpanOperations.MapTo<LeakyReLUOp, Weight>(alpha, input.AsSpan(), output.AsSpan());
        return output;
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
    public static int GetOutputSize(Matrix left, Vector right) => left.ColumnCount;
    public static Vector ForwardTo(Empty _, Matrix left, Vector right, ArrayHandle outputHandle)
    {
        var output = Vector.Of(left.ColumnCount, outputHandle);
        left.MultiplyTo(right, output);
        return output;
    }

    public static void BackwardTo(Empty _, Matrix left, Vector right, Vector output, Vector outputGradient, Matrix leftGradient, Vector rightGradient)
    {
        VectorHelper.MultiplyToMatrixTo(outputGradient, right, leftGradient);
        left.MultiplyTransposedTo(outputGradient, rightGradient);
    }

}

public readonly struct Empty;