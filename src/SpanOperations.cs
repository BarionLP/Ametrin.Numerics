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

    [Obsolete]
    public static void MapTo(ReadOnlySpan<Weight> values, Span<Weight> destination, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap)
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
            SimdVectorHelper.StoreUnsafe(simdMap.Invoke(simdVector), ref destinationPtr, index);
        }

        for (; index < totalSize; index++)
        {
            destination[(int)index] = fallbackMap.Invoke(values[(int)index]);
        }
    }

    [Obsolete]
    public static void MapTo(ReadOnlySpan<Weight> left, ReadOnlySpan<Weight> right, Span<Weight> destination, Func<SimdVector, SimdVector, SimdVector> simdMap, Func<Weight, Weight, Weight> fallbackMap)
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
            SimdVectorHelper.StoreUnsafe(simdMap.Invoke(leftVector, rightVector), ref destinationPtr, index);
        }

        for (; index < totalSize; index++)
        {
            destination[(int)index] = fallbackMap.Invoke(left[(int)index], right[(int)index]);
        }
    }

    public static void MapTo<TOperator>(in TOperator state, ReadOnlySpan<Weight> values, Span<Weight> destination)
        where TOperator : IUnaryOperator<TOperator>
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
        where TOperator : IBinaryOperator<TOperator>
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

}

public interface IBinaryOperator<TState>
{
    static abstract Weight Invoke(in TState state, Weight left, Weight right);
    static abstract SimdVector Invoke(in TState state, SimdVector left, SimdVector right);
}

public interface IUnaryOperator<TState>
{
    static abstract Weight Invoke(in TState state, Weight value);
    static abstract SimdVector Invoke(in TState state, SimdVector value);
}