
// using System.Buffers;

// namespace Ametrin.Numerics.Operations;

// public interface IUnaryTensorOperator<TState, TTensor>
//     where TState : allows ref struct
//     where TTensor : ITensorLike<TTensor>, allows ref struct
// {
//     public abstract static int GetOutputSize(TTensor input);
//     public abstract static TTensor ForwardTo(TState state, TTensor input, StorageHandle outputHandle);
//     public abstract static void BackwardTo(TState state, TTensor input, TTensor output, TTensor outputGradient, TTensor inputGradient);
// }

// public interface IBinaryTensorOperator<TState, TLeftTensor, TRightTensor, TOutputTensor>
//     where TState : allows ref struct
//     where TLeftTensor : allows ref struct
//     where TRightTensor : allows ref struct
//     where TOutputTensor : allows ref struct
// {
//     public abstract static TOutputTensor ForwardTo(TState state, TLeftTensor left, TRightTensor right, StorageHandle outputHandle);
//     public abstract static void BackwardTo(TState state, TLeftTensor left, TRightTensor right, TOutputTensor output, TOutputTensor outputGradient, TLeftTensor leftGradient, TRightTensor rightGradient);
// }

// public readonly ref struct AddTensorsOperator : IBinaryTensorOperator<Empty, Span<Weight>, Span<Weight>, Span<Weight>>
// {
//     public static Span<Weight> ForwardTo(Empty _, Span<Weight> left, Span<Weight> right, StorageHandle outputHandle)
//     {
//         var output = outputHandle.AsSpan()[..left.Length];
//         TensorPrimitives.Add(left, right, output);
//         return output;
//     }

//     public static void BackwardTo(Empty _, Span<Weight> left, Span<Weight> right, Span<Weight> output, Span<Weight> outputGradient, Span<Weight> leftGradient, Span<Weight> rightGradient)
//     {
//         outputGradient.CopyTo(leftGradient);
//         outputGradient.CopyTo(rightGradient);
//     }

//     public static void Test()
//     {
//         Span<Weight> input = default;
//         Span<Weight> softmaxed = default;
//         Span<Weight> relued = default;
//         Span<Weight> output = default;

//         SoftMaxOperation.ForwardTo(default, input, softmaxed);
//         LeakyReLUOperation.ForwardTo(0.1f, input, relued);
//         AddTensorsOperator.ForwardTo(default, softmaxed, relued, output);

//         Span<Weight> gradient = default;
//         Span<Weight> softmaxedGradient = default;
//         Span<Weight> reluedGradient = default;
//         Span<Weight> inputGradient = default;
//         Span<Weight> inputGradientB = default;

//         AddTensorsOperator.BackwardTo(default, softmaxed, relued, output, gradient, softmaxedGradient, reluedGradient);
//         LeakyReLUOperation.BackwardTo(0.1f, input, relued, reluedGradient, inputGradientB);
//         SoftMaxOperation.BackwardTo(default, input, softmaxed, softmaxedGradient, inputGradient);

//         AddTensorsOperator.ForwardTo(default, inputGradient, inputGradientB, inputGradient);
//     }
// }

// public interface IUnaryOperation<TTensor>
//     where TTensor : ITensorLike<TTensor>
// {
//     public TTensor Evaluate();
//     public void Backward(TTensor outputGradient);
// }

// public readonly struct VectorSource : IUnaryOperation<Vector>
// {
//     public required Vector Vector { get; init; }
//     public Vector AccumulatedGradients { get; }

//     public Vector Evaluate() => Vector;
//     public void Backward(Vector outputGradient)
//     {
//         AccumulatedGradients.Add(outputGradient);
//     }
// }

// public struct ChainOperation<TOperator, TState, TTensor> : IUnaryOperation<TTensor>
//     where TOperator : IUnaryTensorOperator<TState, TTensor>, allows ref struct
//     where TTensor : ITensorLike<TTensor>
// {

//     public required IUnaryOperation<TTensor> Source { get; init; }
//     public required TState State { get; init; }

//     private TTensor input;
//     private (TTensor, StorageHandle) output;

//     public TTensor Evaluate()
//     {
//         input = Source.Evaluate();
//         var outputSize = TOperator.GetOutputSize(input);
//         var outputHandle = ArrayPool<Weight>.Shared.RentNumerics(outputSize);
//         var output = TOperator.ForwardTo(State, input, outputHandle);

//         this.output = (output, outputHandle);
//         return output;
//     }

//     public void Backward(TTensor outputGradient)
//     {
//         var (output, outputHandle) = this.output;
//         this.output = default;

//         using var inputGradientHandle = ArrayPool<Weight>.Shared.RentNumerics(input.FlatCount);
//         var inputGradient = TTensor.OfSize(input, inputGradientHandle);
        
//         TOperator.BackwardTo(State, input, output, outputGradient, inputGradient);
//         outputHandle.Dispose();

//         Source.Backward(inputGradient);
//     }
// }

// public readonly ref struct SoftMaxOperation : IUnaryTensorOperator<Empty, Span<Weight>>
// {
//     public static int GetOutputSize(Span<Weight> input) => input.Length;
//     public static void ForwardTo(Empty _, Span<Weight> input, Span<Weight> output)
//     {
//         TensorPrimitives.SoftMax(input, output);
//     }

//     public static void BackwardTo(Empty _, Span<Weight> input, Span<Weight> output, Span<Weight> outputGradient, Span<Weight> inputGradient)
//     {
//         var dot = TensorPrimitives.Dot(output, outputGradient);
//         TensorPrimitives.AddMultiply(outputGradient, -dot, output, inputGradient);
//         // TensorPrimitives.Subtract(outputGradient, dot, inputGradient);
//         // TensorPrimitives.Multiply(inputGradient, output, inputGradient);
//     }
// }

// public readonly ref struct LeakyReLUOperation : IUnaryTensorOperator<Weight, Span<Weight>>
// {
//     public static int GetOutputSize(Span<Weight> input) => input.Length;
//     public static void ForwardTo(Weight alpha, Span<Weight> input, Span<Weight> output)
//     {
//         SpanOperations.MapTo<LeakyReLUOp, Weight>(alpha, input, output);
//     }

//     public static void BackwardTo(Weight alpha, Span<Weight> input, Span<Weight> output, Span<Weight> outputGradient, Span<Weight> inputGradient)
//     {
//         SpanOperations.MapTo<LeakyReLUGradientOp, Weight>(alpha, input, outputGradient, inputGradient);
//     }

//     private readonly ref struct LeakyReLUOp : IUnaryOperator<Weight>
//     {
//         public static Weight Invoke(in Weight alpha, Weight input) => input > 0 ? input : alpha * input;
//         public static SimdVector Invoke(in Weight alpha, SimdVector input)
//             => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), input, input * alpha);
//     }

//     private readonly ref struct LeakyReLUGradientOp : IBinaryOperator<Weight>
//     {
//         public static Weight Invoke(in Weight alpha, Weight input, Weight gradient) => input > 0 ? gradient : gradient * alpha;
//         public static SimdVector Invoke(in Weight alpha, SimdVector input, SimdVector gradient)
//             => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), gradient, gradient * alpha);
//     }
// }

// public readonly ref struct Empty;