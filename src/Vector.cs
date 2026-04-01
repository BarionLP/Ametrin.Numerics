using System.ComponentModel;
using System.Runtime.CompilerServices; // required in release
using System.Runtime.InteropServices;
using System.Text;

namespace Ametrin.Numerics;

// Memory<Weight> exposes no direct ref accessor, only via Span
public readonly struct Vector : ITensorLike<Vector>
{
    private readonly int startIndex;
    private readonly ArrayHandle source;
    public int Count { get; }

    private Vector(ArrayHandle source, int start, int count)
    {
#if DEBUG
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(checked(start + count), source.Length);
#endif
        startIndex = start;
        this.source = source;
        Count = count;
    }

    public ref Weight this[int index]
    {
#if !DEBUG
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        get
        {
            Debug.Assert(index < Count);
            Debug.Assert(!source.IsDisposed);
            return ref source.Array[startIndex + index];
        }
    }

    public ref Weight this[nuint index]
    {
#if !DEBUG
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        get
        {
            Debug.Assert(index < (nuint)Count);
            Debug.Assert(!source.IsDisposed);
            return ref source.Array[(nuint)startIndex + index];
        }
    }


    public Vector Slice(int start, int count)
    {
#if DEBUG
        Debug.Assert(!source.IsDisposed);
        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(start + count, Count);
#endif

        return new(source, startIndex + start, count);
    }


    [EditorBrowsable(EditorBrowsableState.Never)]
    public int FlatCount => Count;


    public Span<Weight> AsSpan() => source.AsSpan(startIndex, Count);

    public override string ToString()
    {
        Debug.Assert(!source.IsDisposed);
        var builder = new StringBuilder("[");
        var endIndex = startIndex + Count;
        for (int i = startIndex; i < endIndex; i++)
        {
            if (i > startIndex) builder.Append(' ');
            builder.Append(source.Array[i].ToString("+0.00;-0.00;+0.00"));
        }
        builder.Append(']');
        return builder.ToString();
    }

    public static Vector Empty { get; } = new(new([], null), 0, 0);
    public static Vector Create(int size) => new(new(new Weight[size], null), 0, size);
    public static Vector Of(Weight[] array) => new(new(array, null), 0, array.Length);
    public static Vector Of(Weight[] array, int size)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(size, array.Length);
        return new Vector(new(array, null), 0, size);
    }
    public static Vector Of(ArrayHandle handle, int size)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(size, handle.Length);
        Debug.Assert(!handle.IsDisposed);
        return new Vector(handle, 0, size);
    }
    public static Vector OfSize(Vector template) => Create(template.Count);
    public static Vector OfSize(Vector template, ArrayHandle handle) => Of(handle, template.Count);
}

[NumericsHelper<Vector>(GenerateFromTensorPrimitives = [nameof(TensorPrimitives.Add), nameof(TensorPrimitives.Subtract)])]
public static partial class VectorHelper
{
    public static Weight Sum(this Vector vector) => TensorPrimitives.Sum(vector.AsSpan());
    public static Weight Magnitude(this Vector vector) => Weight.Sqrt(TensorPrimitives.SumOfSquares(vector.AsSpan()));

    public static Weight MaxMagnitude(this Vector vector) => TensorPrimitives.MaxMagnitude(vector.AsSpan());

    public static Weight Dot(this Vector left, Vector right)
    {
        NumericsDebug.AssertSameDimensions(left, right);
        return TensorPrimitives.Dot(left.AsSpan(), right.AsSpan());
    }

    [GenerateVariants]
    public static void PointwiseExpTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        TensorPrimitives.Exp(vector.AsSpan(), destination.AsSpan());
    }

    [GenerateVariants]
    public static void PointwiseLogTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        TensorPrimitives.Log(vector.AsSpan(), destination.AsSpan());
    }

    [GenerateVariants]
    public static void SoftMaxTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);

        var max = vector.Max();
        vector.SubtractPointwiseTo(max, destination);
        destination.PointwiseExpToSelf();
        var sum = destination.Sum();
        destination.DivideToSelf(sum);

        // still minimally slower because it actually divides instead of multiply by 1/sum (.NET 10)
        // TensorPrimitives.SoftMax(destination.AsSpan(), destination.AsSpan());
        NumericsDebug.AssertValidNumbers(destination);
    }

    [GenerateVariants]
    public static void SigmoidTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        TensorPrimitives.Sigmoid(vector.AsSpan(), destination.AsSpan());
        NumericsDebug.AssertValidNumbers(destination);
    }

    [GenerateVariants]
    public static void SwishTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        ref var destinationPtr = ref MemoryMarshal.GetReference(destination.AsSpan());
        var dataSize = (nuint)SimdVector.Count;
        var totalSize = (nuint)vector.Count;

        nuint index = 0;
        for (; index + dataSize <= totalSize; index += dataSize)
        {
            var simdVector = SimdVectorHelper.LoadUnsafe(ref vectorPtr, index);
            SimdVectorHelper.StoreUnsafe(simdVector / (SimdVector.One + SimdVectorHelper.Exp(-simdVector)), ref destinationPtr, index);
        }

        for (; index < totalSize; index++)
        {
            var x = vector[index];
            destination[index] = x / (1f + Weight.Exp(-x));
        }
    }

    [GenerateVariants]
    public static void MapTo(this Vector vector, Func<Weight, Weight> map, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        SpanOperations.MapTo(vector.AsSpan(), destination.AsSpan(), map);
    }

    [GenerateVariants]
    public static void MapTo<TOperator>(this Vector vector, in TOperator state, Vector destination)
        where TOperator : IUnaryOperator<TOperator>
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        SpanOperations.MapTo(state, vector.AsSpan(), destination.AsSpan());
    }

    public static void MapToFirst(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map) => vectors.MapTo(map, vectors.a);
    public static Vector Map(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map)
    {
        var destination = Vector.Create(vectors.a.Count);
        vectors.MapTo(map, destination);
        return destination;
    }
    public static void MapTo(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vectors.a, vectors.b, destination);
        SpanOperations.MapTo(vectors.a.AsSpan(), vectors.b.AsSpan(), destination.AsSpan(), map);
    }
    public static void MapToFirst<TOperator>(this (Vector a, Vector b) vectors, in TOperator state) where TOperator : IBinaryOperator<TOperator> => vectors.MapTo<TOperator>(state, vectors.a);
    public static Vector Map<TOperator>(this (Vector a, Vector b) vectors, in TOperator state)
        where TOperator : IBinaryOperator<TOperator>
    {
        var destination = Vector.Create(vectors.a.Count);
        vectors.MapTo(state, destination);
        return destination;
    }
    public static void MapTo<TOperator>(this (Vector a, Vector b) vectors, in TOperator state, Vector destination)
        where TOperator : IBinaryOperator<TOperator>
    {
        NumericsDebug.AssertSameDimensions(vectors.a, vectors.b, destination);
        SpanOperations.MapTo(state, vectors.a.AsSpan(), vectors.b.AsSpan(), destination.AsSpan());
    }

    public static Vector Map(this (Vector a, Vector b, Vector c) vectors, Func<Weight, Weight, Weight, Weight> map)
    {
        var result = Vector.Create(vectors.a.Count);
        vectors.MapTo(map, result);
        return result;
    }
    public static void MapTo(this (Vector a, Vector b, Vector c) vectors, Func<Weight, Weight, Weight, Weight> map, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vectors.a, vectors.b, vectors.c, destination);
        SpanOperations.MapTo(vectors.a.AsSpan(), vectors.b.AsSpan(), vectors.c.AsSpan(), destination.AsSpan(), map);
    }

    [GenerateVariants]
    public static void PointwiseMultiplyTo(this Vector left, Vector right, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        TensorPrimitives.Multiply(left.AsSpan(), right.AsSpan(), destination.AsSpan());
    }

    public static void PointwiseMultiplyAddTo(this Vector left, Vector right, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        TensorPrimitives.MultiplyAdd(left.AsSpan(), right.AsSpan(), destination.AsSpan(), destination.AsSpan());
    }

    public static Vector Multiply(this Vector vector, Matrix matrix)
    {
        var result = Vector.Create(matrix.ColumnCount);
        vector.MultiplyTo(matrix, result);
        return result;
    }

    public static void MultiplyTo(this Vector vector, Matrix matrix, Vector destination)
    {
        // v*M is numerically equivalent to M^T*v 
        destination.ResetZero();
        MultiplyAddTo(vector, matrix, destination);
    }
    public static void MultiplyAddTo(this Vector vector, Matrix matrix, Vector destination)
    {
        // Story time: swapping loops increased performance by 85 % because of increased cache hits (before simd impl)
        Debug.Assert(vector.Count == matrix.RowCount);
        Debug.Assert(destination.Count == matrix.ColumnCount);

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(destination.AsSpan());
        var dataSize = (nuint)SimdVector.Count;
        var rowCount = (nuint)matrix.RowCount;
        var columnCount = (nuint)matrix.ColumnCount;

        // computes d[column] += v[row] * M[row, column] for each cell 
        for (nuint row = 0; row < rowCount; row++)
        {
            var rowValue = new SimdVector(vector[row]);
            var rowOffset = row * columnCount;
            nuint column = 0;
            for (; column + dataSize <= columnCount; column += dataSize)
            {
                var resultValues = SimdVectorHelper.LoadUnsafe(ref resultPtr, column);
                var matrixValues = SimdVectorHelper.LoadUnsafe(ref matrixPtr, rowOffset + column);
                resultValues += rowValue * matrixValues;

                SimdVectorHelper.StoreUnsafe(resultValues, ref resultPtr, column);
            }

            for (; column < columnCount; column++)
            {
                destination[column] += vector[row] * matrix[rowOffset + column];
            }
        }
    }

    public static Matrix MultiplyToMatrix(Vector rowVector, Vector columnVector)
    {
        var result = Matrix.Create(rowVector.Count, columnVector.Count);
        MultiplyToMatrixTo(rowVector, columnVector, result);
        return result;
    }

    public static void MultiplyToMatrixTo(Vector rowVector, Vector columnVector, Matrix destination)
    {
        Debug.Assert(rowVector.Count == destination.RowCount);
        Debug.Assert(columnVector.Count == destination.ColumnCount);

        ref var columnPtr = ref MemoryMarshal.GetReference(columnVector.AsSpan());
        ref var destinationPtr = ref MemoryMarshal.GetReference(destination.AsSpan());

        var rowCount = (nuint)rowVector.Count;
        var columnCount = (nuint)columnVector.Count;

        var dataSize = (nuint)SimdVector.Count;

        for (nuint row = 0; row < rowCount; row++)
        {
            var rowValue = new SimdVector(rowVector[row]);
            var rowOffset = row * columnCount;

            nuint column = 0;
            for (; column + dataSize <= columnCount; column += dataSize)
            {
                var columnValues = SimdVectorHelper.LoadUnsafe(ref columnPtr, column);
                var destinationValues = rowValue * columnValues;

                SimdVectorHelper.StoreUnsafe(destinationValues, ref destinationPtr, rowOffset + column);
            }

            for (; column < columnCount; column++)
            {
                destination[rowOffset + column] = rowVector[row] * columnVector[column];
            }
        }
    }

    public static void MultiplyToMatrixAddTo(Vector rowVector, Vector columnVector, Matrix destination)
    {
        Debug.Assert(rowVector.Count == destination.RowCount);
        Debug.Assert(columnVector.Count == destination.ColumnCount);

        ref var columnPtr = ref MemoryMarshal.GetReference(columnVector.AsSpan());
        ref var destinationPtr = ref MemoryMarshal.GetReference(destination.AsSpan());

        var rowCount = (nuint)rowVector.Count;
        var columnCount = (nuint)columnVector.Count;

        var dataSize = (nuint)SimdVector.Count;

        for (nuint row = 0; row < rowCount; row++)
        {
            var rowValue = new SimdVector(rowVector[row]);
            var rowOffset = row * columnCount;

            nuint column = 0;
            for (; column + dataSize <= columnCount; column += dataSize)
            {
                var columnValues = SimdVectorHelper.LoadUnsafe(ref columnPtr, column);
                var destinationValues = SimdVectorHelper.LoadUnsafe(ref destinationPtr, rowOffset + column);
                destinationValues += rowValue * columnValues;

                SimdVectorHelper.StoreUnsafe(destinationValues, ref destinationPtr, rowOffset + column);
            }

            for (; column < columnCount; column++)
            {
                destination[rowOffset + column] += rowVector[row] * columnVector[column];
            }
        }
    }

    [GenerateVariants]
    public static void MultiplyTo(this Vector vector, Weight factor, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        TensorPrimitives.Multiply(vector.AsSpan(), factor, destination.AsSpan());
    }

    [GenerateVariants]
    public static void MultiplyAddTo(this Vector vector, Weight factor, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        TensorPrimitives.MultiplyAdd(vector.AsSpan(), factor, destination.AsSpan(), destination.AsSpan());
    }

    [GenerateVariants]
    public static void DivideTo(this Vector vector, Weight divisor, Vector destination)
    {
        // TODO: caller should decide this because this is only good when vector and divisor are small
        MultiplyTo(vector, 1 / divisor, destination);
    }

    [GenerateVariants]
    public static void SubtractPointwiseTo(this Vector left, Weight right, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(left, destination);
        TensorPrimitives.Subtract(left.AsSpan(), right, destination.AsSpan());
    }

    public static int MaximumIndex(this Vector vector)
    {
        return TensorPrimitives.IndexOfMax(vector.AsSpan());
    }

    public static Weight Max(this Vector vector) => TensorPrimitives.Max(vector.AsSpan());
    public static Weight Min(this Vector vector) => TensorPrimitives.Min(vector.AsSpan());

    public static Vector CreateCopy(this Vector vector)
    {
        var copy = Vector.Create(vector.Count);
        vector.AsSpan().CopyTo(copy.AsSpan());
        return copy;
    }

    public static void CopyTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        vector.AsSpan().CopyTo(destination.AsSpan());
    }

    public static void ResetZero(this Vector vector) => vector.AsSpan().Clear();
    public static void Fill(this Vector vector, Weight value) => vector.AsSpan().Fill(value);
    public static void Fill(this Vector vector, Func<Weight> factory)
    {
        var span = vector.AsSpan();
        for (var i = 0; i < span.Length; i++)
        {
            span[i] = factory();
        }
    }
}
