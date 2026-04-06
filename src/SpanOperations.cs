using System.Runtime.InteropServices;

namespace Ametrin.Numerics;

public static class SpanOperations
{
    public static void MapTo<T>(ReadOnlySpan<T> values, Span<T> destination, Func<T, T> map)
    {
        Debug.Assert(values.Length == destination.Length);
        for (int i = 0; i < values.Length; i++)
        {
            destination[i] = map(values[i]);
        }
    }

    public static void MapTo<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> destination, Func<T, T, T> map)
    {
        Debug.Assert(left.Length == right.Length && left.Length == destination.Length);

        for (int i = 0; i < left.Length; i++)
        {
            destination[i] = map(left[i], right[i]);
        }
    }
    public static void MapTo<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, ReadOnlySpan<T> c, Span<T> destination, Func<T, T, T, T> map)
    {
        Debug.Assert(a.Length == b.Length && a.Length == c.Length && a.Length == destination.Length);

        for (int i = 0; i < a.Length; i++)
        {
            destination[i] = map(a[i], b[i], c[i]);
        }
    }

    public static void MapTo<TOperator>(in TOperator state, ReadOnlySpan<Weight> values, Span<Weight> destination)
        where TOperator : IUnaryOperator<TOperator>, allows ref struct
        => MapTo<TOperator, TOperator>(state, values, destination);
    public static void MapTo<TOperator, TState>(in TState state, ReadOnlySpan<Weight> values, Span<Weight> destination)
        where TOperator : IUnaryOperator<TState>, allows ref struct
        where TState : allows ref struct
    {
        NumericsDebug.AssertSameDimensions(values, destination);
        ref var vectorPtr = ref MemoryMarshal.GetReference(values);
        ref var destinationPtr = ref MemoryMarshal.GetReference(destination);
        var dataSize = (nuint)SimdVector.Count;
        var totalSize = (nuint)values.Length;

        nuint index = 0;
        for (; index + dataSize <= totalSize; index += dataSize)
        {
            var simdVector = SimdVectorHelper.LoadUnsafe(ref vectorPtr, index);
            SimdVectorHelper.StoreUnsafe(TOperator.Invoke(in state, simdVector), ref destinationPtr, index);
        }

        // significantly slower :(
        // if (index < totalSize)
        // {
        //     var remainingCount = (int)(totalSize - index);
        //     Span<Weight> tmp = stackalloc Weight[SimdVector.Count];
        //     ref var tmpPtr = ref MemoryMarshal.GetReference(tmp);

        //     values[(int)index..].CopyTo(tmp);
        //     var simdVector = SimdVectorHelper.LoadUnsafe(ref tmpPtr);

        //     SimdVectorHelper.StoreUnsafe(TOperator.Invoke(simdVector), ref tmpPtr);
        //     tmp[..remainingCount].CopyTo(destination[(int)index..]);
        // }

        for (; index < totalSize; index++)
        {
            destination[(int)index] = TOperator.Invoke(state, values[(int)index]);
        }
    }

    public static void MapTo<TOperator>(in TOperator state, ReadOnlySpan<Weight> left, ReadOnlySpan<Weight> right, Span<Weight> destination)
        where TOperator : IBinaryOperator<TOperator>, allows ref struct
        => MapTo<TOperator, TOperator>(state, left, right, destination);

    public static void MapTo<TOperator, TState>(in TState state, ReadOnlySpan<Weight> left, ReadOnlySpan<Weight> right, Span<Weight> destination)
        where TOperator : IBinaryOperator<TState>, allows ref struct
        where TState : allows ref struct
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        ref var leftPtr = ref MemoryMarshal.GetReference(left);
        ref var rightPtr = ref MemoryMarshal.GetReference(right);
        ref var destinationPtr = ref MemoryMarshal.GetReference(destination);
        var dataSize = (nuint)SimdVector.Count;
        var totalSize = (nuint)left.Length;

        nuint index = 0;
        for (; index + dataSize <= totalSize; index += dataSize)
        {
            var leftVector = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
            var rightVector = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
            SimdVectorHelper.StoreUnsafe(TOperator.Invoke(state, leftVector, rightVector), ref destinationPtr, index);
        }

        for (; index < totalSize; index++)
        {
            destination[(int)index] = TOperator.Invoke(state, left[(int)index], right[(int)index]);
        }
    }

    public static void MapTo<TOperator>(in TOperator state, ReadOnlySpan<Weight> first, ReadOnlySpan<Weight> second, ReadOnlySpan<Weight> third, Span<Weight> destination)
        where TOperator : ITernaryOperator<TOperator>
    {
        NumericsDebug.AssertSameDimensions(first, second, third, destination);
        ref var firstPtr = ref MemoryMarshal.GetReference(first);
        ref var secondPtr = ref MemoryMarshal.GetReference(second);
        ref var thirdPtr = ref MemoryMarshal.GetReference(third);
        ref var destinationPtr = ref MemoryMarshal.GetReference(destination);
        var dataSize = (nuint)SimdVector.Count;
        var totalSize = (nuint)first.Length;

        nuint index = 0;
        for (; index + dataSize <= totalSize; index += dataSize)
        {
            var firstVector = SimdVectorHelper.LoadUnsafe(ref firstPtr, index);
            var secondVector = SimdVectorHelper.LoadUnsafe(ref secondPtr, index);
            var thirdVector = SimdVectorHelper.LoadUnsafe(ref thirdPtr, index);
            SimdVectorHelper.StoreUnsafe(TOperator.Invoke(state, firstVector, secondVector, thirdVector), ref destinationPtr, index);
        }

        for (; index < totalSize; index++)
        {
            destination[(int)index] = TOperator.Invoke(state, first[(int)index], second[(int)index], third[(int)index]);
        }
    }

}

public interface IUnaryOperator<TState>
    where TState : allows ref struct
{
    static abstract Weight Invoke(in TState state, Weight value);
    static abstract SimdVector Invoke(in TState state, SimdVector value);
}

public interface IBinaryOperator<TState>
    where TState : allows ref struct
{
    static abstract Weight Invoke(in TState state, Weight left, Weight right);
    static abstract SimdVector Invoke(in TState state, SimdVector left, SimdVector right);
}

public interface ITernaryOperator<TState>
    where TState : allows ref struct
{
    static abstract Weight Invoke(in TState state, Weight first, Weight second, Weight third);
    static abstract SimdVector Invoke(in TState state, SimdVector first, SimdVector second, SimdVector third);
}

public readonly ref struct TanhOperator : IUnaryOperator<Empty>
{
    public static Weight Invoke(in Empty state, Weight value) => Weight.Tanh(value);

    public static SimdVector Invoke(in Empty state, SimdVector value)
    {
        var abs = SimdVectorHelper.Abs(value);
        var z = SimdVectorHelper.Exp(-2 * abs) - SimdVector.One; // may be numerically unstable there is a internal ExpM1 operation in Numerics.Tensors but vectorized it just falls back to this
        var z2 = -z / (z + SimdVectorHelper.Create<Weight>(2));
        var sign = SimdVectorHelper.As<float, uint>(value) & SimdVectorHelper.Create(~(uint)int.MaxValue); // when changing Weight also change uint to match the binary size
        return SimdVectorHelper.As<uint, float>(sign ^ SimdVectorHelper.As<float, uint>(z2));
    }
}

public readonly ref struct Empty;