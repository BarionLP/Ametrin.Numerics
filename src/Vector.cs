using System.Runtime.InteropServices;
using System.Text;

namespace Ametrin.Numerics;

public interface Vector
{
    public static readonly Vector Empty = new VectorSlice([], 0, 0);
    public int Count { get; }
    public ref Weight this[int index] { get; }
    public ref Weight this[nuint index] { get; }
    public Vector Slice(int index, int count);
    public Span<Weight> AsSpan();

    public static Vector Create(int size) => new VectorSlice(new Weight[size], 0, size);
    public static Vector Of(Weight[] array) => new VectorSlice(array, 0, array.Length);
    public static Vector Of(int size, Weight[] array)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(size, array.Length);
        return new VectorSlice(array, 0, size);
    }
    public static Vector OfSize(Vector template) => Create(template.Count);
}

internal readonly struct VectorSlice(Weight[] _source, int start, int count) : Vector
{
    private readonly int _startIndex = start >= 0 && start + count <= _source.Length ? start : throw new ArgumentOutOfRangeException(nameof(start), "slice is out of range");
    private readonly Weight[] _source = _source;

    public ref Weight this[int index] => ref _source[_startIndex + index];

    public ref Weight this[nuint index] => ref _source[_startIndex + (int)index];

    public Vector Slice(int index, int count)
    {
#if DEBUG
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(index + count, Count);
#endif
        return new VectorSlice(_source, _startIndex + index, count);
    }

    public int Count { get; } = count;

    public Span<Weight> AsSpan() => _source.AsSpan(_startIndex, Count);

    public override string ToString()
    {
        var builder = new StringBuilder("[");
        var data = AsSpan();
        for (int i = 0; i < data.Length; i++)
        {
            if (i > 0) builder.Append(' ');
            builder.Append(data[i].ToString("+0.00;-0.00;+0.00"));
        }
        builder.Append(']');
        return builder.ToString();
    }
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
        return TensorPrimitives.Dot<Weight>(left.AsSpan(), right.AsSpan());
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

        // was slower in .net 9.preview.7
        // TensorPrimitives.SoftMax(vector.AsSpan(), destination.AsSpan());
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
            destination[index] = x / (1f + MathF.Exp(-x));
        }

        // for (var i = 0; i < vector.Count; i++)
        // {
        //     var x = vector[i];
        //     destination[i] = x / (1f + MathF.Exp(-x));
        // }
    }

    [GenerateVariants]
    public static void MapTo(this Vector vector, Func<Weight, Weight> map, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        SpanOperations.MapTo(vector.AsSpan(), destination.AsSpan(), map);
    }

    [GenerateVariants]
    public static void MapTo(this Vector vector, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap, Vector destination)
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
            SimdVectorHelper.StoreUnsafe(simdMap.Invoke(simdVector), ref destinationPtr, index);
        }

        for (; index < totalSize; index++)
        {
            destination[index] = fallbackMap.Invoke(vector[index]);
        }
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
        destination.ResetZero();
        MultiplyAddTo(vector, matrix, destination);
    }
    public static void MultiplyAddTo(this Vector vector, Matrix matrix, Vector destination)
    {
        //Story time: swapping loops increased performance by 85 % because of increased cache hits (before simd impl)
        Debug.Assert(vector.Count == matrix.RowCount);
        Debug.Assert(destination.Count == matrix.ColumnCount);

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(destination.AsSpan());
        var dataSize = (nuint)SimdVector.Count;
        var rowCount = (nuint)matrix.RowCount;
        var columnCount = (nuint)matrix.ColumnCount;

        // computes d[column] += v[row] * M[row, column] foreach row column pair 
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
        // var maxIndex = 0;
        // for (int i = 1; i < vector.Count; i++)
        // {
        //     if (vector[i] > vector[maxIndex])
        //     {
        //         maxIndex = i;
        //     }
        // }
        // return maxIndex;
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

    public static void Fill(this Vector vector, Weight value) => vector.AsSpan().Fill(value);
    public static void ResetZero(this Vector vector) => vector.AsSpan().Clear();
}
