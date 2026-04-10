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
        var softmaxed = new UnaryOperationNode<SoftMaxOperation, Empty, Vector> { Source = input, OperationState = default };
        var relued = new UnaryOperationNode<LeakyReLUOperation, float, Vector> { Source = input, OperationState = 0.1f };
        var output = new BinaryOperationNode<AddTensorsOperator<Vector>, Empty, Vector, Vector, Vector> { LeftSource = softmaxed, RightSource = relued, OperationState = default, };

        var context = new GraphExecutionContext();

        output.Backward(context, output.Forward(context));
    }
}

public interface IOperationNode
{
    public object CreateState();
}

public interface IOperationNode<TTensor> : IOperationNode
    where TTensor : struct, ITensorLike<TTensor>
{
    public TTensor Forward(GraphExecutionContext context);
    public void Backward(GraphExecutionContext context, TTensor outputGradient);
}

public sealed class InputNode<TTensor> : IOperationNode<TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    private TTensor input = TTensor.Empty;

    public void Set(TTensor input)
    {
        this.input = input;
    }

    public TTensor Forward(GraphExecutionContext context)
    {
        return input;
    }
    public void Backward(GraphExecutionContext context, TTensor outputGradient) { }

    public object CreateState() => throw new NotImplementedException();
}

public sealed class LearnableWeightsNode<TTensor>(TTensor weights) : IOperationNode<TTensor>
    where TTensor : struct, ITensorLike<TTensor>
{
    private readonly TTensor weights = weights;

    public TTensor Forward(GraphExecutionContext context) => weights;
    public void Backward(GraphExecutionContext context, TTensor outputGradient)
    {
        var gradients = context.GetOrCreate<State>(this).Gradients.AsSpan();
        TensorPrimitives.Add(gradients, outputGradient.AsSpan(), gradients);
    }

    public object CreateState() => new State(weights);

    private sealed class State(TTensor template)
    {
        public readonly TTensor Gradients = TTensor.OfSize(template);
    }
}

public sealed class UnaryOperationNode<TOperator, TState, TTensor> : IOperationNode<TTensor>
    where TOperator : IUnaryTensorOperator<TState, TTensor>, allows ref struct
    where TTensor : struct, ITensorLike<TTensor>
{

    public required IOperationNode<TTensor> Source { get; init; }
    public required TState OperationState { get; init; }

    public TTensor Forward(GraphExecutionContext context)
    {
        var state = context.GetOrCreate<State>(this);
        state.input = Source.Forward(context);
        TOperator.ForwardTo(OperationState, state.input, state.output);
        return state.output;
    }

    public void Backward(GraphExecutionContext context, TTensor outputGradient)
    {
        var state = context.GetOrCreate<State>(this);
        state.inputGradient.SetSize(state.input);
        TOperator.BackwardTo(OperationState, state.input, state.output, outputGradient, state.inputGradient);
        Source.Backward(context, state.inputGradient);
    }

    public object CreateState() => new State();

    private sealed class State
    {
        public TTensor input;
        public readonly Dynamic<TTensor> output = new();
        public readonly Dynamic<TTensor> inputGradient = new();
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
    public required TState OperationState { get; init; }

    public TOutputTensor Forward(GraphExecutionContext context)
    {
        var state = context.GetOrCreate<State>(this);
        state.inputLeft = LeftSource.Forward(context);
        state.inputRight = RightSource.Forward(context);
        TOperator.ForwardTo(OperationState, state.inputLeft, state.inputRight, state.output);
        return state.output;
    }

    public void Backward(GraphExecutionContext context, TOutputTensor outputGradient)
    {
        var state = context.GetOrCreate<State>(this);
        state.leftGradientHandle.SetSize(state.inputLeft);
        state.rightGradientHandle.SetSize(state.inputRight);

        var leftGradient = state.leftGradientHandle.Tensor;
        var rightGradient = state.rightGradientHandle.Tensor;

        TOperator.BackwardTo(OperationState, state.inputLeft, state.inputRight, state.output, outputGradient, ref leftGradient, ref rightGradient);

        LeftSource.Backward(context, leftGradient);
        RightSource.Backward(context, rightGradient);
    }

    public object CreateState() => new State();

    private sealed class State
    {
        public TLeftTensor inputLeft;
        public TRightTensor inputRight;
        public readonly Dynamic<TOutputTensor> output = new();
        public readonly Dynamic<TLeftTensor> leftGradientHandle = new();
        public readonly Dynamic<TRightTensor> rightGradientHandle = new();
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


public sealed class GraphExecutionContext
{
    private readonly Dictionary<object, object> _nodeState = [];
    public TState GetOrCreate<TState>(IOperationNode node) where TState : class
    {
        if (!_nodeState.TryGetValue(node, out var state))
        {
            state = node.CreateState();
            _nodeState.Add(node, state);
        }

        return (TState)state;
    }

    public int Version { get; set; }
}