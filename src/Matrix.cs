using System.Runtime.InteropServices;
using System.Text;

namespace Ametrin.Numerics;

// must be a row major continuous chunk of memory for current simd to work 
public readonly struct Matrix(int rowCount, int columnCount, Vector storage) : ITensorLike<Matrix>
{
    public Vector Storage { get; } = storage;
    public int RowCount { get; } = rowCount;
    public int ColumnCount { get; } = columnCount;
    public int FlatCount => Storage.Count;

    public ref Weight this[int row, int column] => ref Storage[GetFlatIndex(row, column)];
    public ref Weight this[nuint flatIndex] => ref Storage[flatIndex];
    public ref Weight this[int flatIndex] => ref Storage[flatIndex];
    public Span<Weight> AsSpan() => Storage.AsSpan();


    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Matrix ({RowCount}x{ColumnCount}):");
        for (int i = 0; i < RowCount; i++)
        {
            sb.Append($"{i}: ");
            for (int j = 0; j < ColumnCount; j++)
            {
                sb.Append(this[i, j].ToString("+0.00;-0.00;+0.00")).Append(' ');
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }

    internal int GetFlatIndex(int row, int column)
    {
#if DEBUG
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, RowCount);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, ColumnCount);
#endif

        return row * ColumnCount + column;
    }

    public static Matrix Empty { get; } // = new(0, 0, Vector.Empty);
    public static Matrix CreateSquare(int size) => Create(size, size);
    public static Matrix Create(int rowCount, int columnCount) => new(rowCount, columnCount, Vector.Create(rowCount * columnCount));
    public static Matrix Of(int rowCount, int columnCount, Weight[] storage) => Of(rowCount, columnCount, Vector.Of(storage));
    public static Matrix Of(int rowCount, int columnCount, Vector storage)
    {
        if (storage.Count != columnCount * rowCount)
        {
            throw new ArgumentException("storage size does not match specified dimensions");
        }

        return new(rowCount, columnCount, storage);
    }

    public static Matrix OfSize(Matrix template) => Create(template.RowCount, template.ColumnCount);
    public static Matrix OfSize(Matrix template, Vector storage) => Of(template.RowCount, template.ColumnCount, storage);
}

[NumericsHelper<Matrix>(GenerateFromTensorPrimitives = [nameof(TensorPrimitives.Add), nameof(TensorPrimitives.Subtract)])]
public static partial class MatrixHelper
{
    public static Weight Sum(this Matrix matrix) => TensorPrimitives.Sum<Weight>(matrix.AsSpan());
    public static Weight Max(this Matrix matrix) => TensorPrimitives.Max<Weight>(matrix.AsSpan());
    public static Weight Min(this Matrix matrix) => TensorPrimitives.Min<Weight>(matrix.AsSpan());
    public static Weight MaxMagnitude(this Matrix vector) => TensorPrimitives.MaxMagnitude(vector.AsSpan());

    public static Vector Multiply(this Matrix matrix, Vector vector)
    {
        var destination = Vector.Create(matrix.RowCount);
        MultiplyTo(matrix, vector, destination);
        return destination;
    }

    public static void MultiplyTo(this Matrix matrix, Vector vector, Vector destination)
    {
        Debug.Assert(vector.Count == matrix.ColumnCount);
        Debug.Assert(destination.Count == matrix.RowCount);

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());

        var dataSize = (nuint)SimdVector.Count;
        var rowCount = (nuint)matrix.RowCount;
        var columnCount = (nuint)matrix.ColumnCount;

        for (nuint row = 0; row < rowCount; row++)
        {
            nuint rowOffset = row * columnCount;
            nuint column = 0;
            var aggregator = SimdVector.Zero;
            for (; column + dataSize <= columnCount; column += dataSize)
            {
                var matrixVec = SimdVectorHelper.LoadUnsafe(ref matrixPtr, rowOffset + column);
                var vectorVec = SimdVectorHelper.LoadUnsafe(ref vectorPtr, column);
                aggregator += matrixVec * vectorVec;
            }

            ref var sum = ref destination[row];
            sum = SimdVectorHelper.Sum(aggregator);

            for (; column < columnCount; column++)
            {
                sum += matrix[rowOffset + column] * vector[column];
            }
        }
    }

    public static void MultiplyAddTo(this Matrix matrix, Vector vector, Vector destination)
    {
        Debug.Assert(vector.Count == matrix.ColumnCount);
        Debug.Assert(destination.Count == matrix.RowCount);

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        // ref var destinationPtr = ref MemoryMarshal.GetReference(destination.AsSpan());

        var dataSize = (nuint)SimdVector.Count;
        var rowCount = (nuint)matrix.RowCount;
        var columnCount = (nuint)matrix.ColumnCount;

        for (nuint row = 0; row < rowCount; row++)
        {
            nuint rowOffset = row * columnCount;
            nuint column = 0;
            var aggregator = SimdVector.Zero;
            for (; column + dataSize <= columnCount; column += dataSize)
            {
                var matrixVec = SimdVectorHelper.LoadUnsafe(ref matrixPtr, rowOffset + column);
                var vectorVec = SimdVectorHelper.LoadUnsafe(ref vectorPtr, column);
                aggregator += matrixVec * vectorVec;
            }

            ref var sum = ref destination[row];
            sum += SimdVectorHelper.Sum(aggregator);

            for (; column < columnCount; column++)
            {
                sum += matrix[rowOffset + column] * vector[column];
            }
        }
    }

    public static void MultiplyRowwiseTo(this Matrix left, Matrix right, Matrix destination)
    {
        Debug.Assert(left.RowCount == destination.RowCount);

        for (int rowIndex = 0; rowIndex < left.RowCount; rowIndex++)
        {
            var row = left.RowRef(rowIndex);
            var destinationRow = destination.RowRef(rowIndex);
            right.MultiplyTo(row, destinationRow);
        }
    }
    public static Vector MultiplyTransposed(this Matrix matrix, Vector vector)
    {
        // see MultiplyTransposedTo
        return vector.Multiply(matrix);
    }
    public static void MultiplyTransposedTo(this Matrix matrix, Vector vector, Vector destination)
    {
        // M^T*v is numerically equivalent to v*M
        // M^T*v produces a column vector
        // v*M   produces a row vector
        vector.MultiplyTo(matrix, destination);
    }

    public static void MultiplyTransposedAddTo(this Matrix matrix, Vector vector, Vector destination)
    {
        vector.MultiplyAddTo(matrix, destination);
    }

    [GenerateVariants]
    public static void MapTo(this Matrix matrix, Func<Weight, Weight> map, Matrix destination)
    {
        NumericsDebug.AssertSameDimensions(matrix, destination);
        SpanOperations.MapTo(matrix.AsSpan(), destination.AsSpan(), map);
    }

    public static void MapToFirst(this (Matrix a, Matrix b) matrices, Func<Weight, Weight, Weight> map) => matrices.MapTo(matrices.a, map);
    public static Matrix Map(this (Matrix a, Matrix b) matrices, Func<Weight, Weight, Weight> map)
    {
        var destination = Matrix.Create(matrices.a.RowCount, matrices.a.ColumnCount);
        matrices.MapTo(destination, map);
        return destination;
    }
    public static void MapTo(this (Matrix a, Matrix b) matrices, Matrix destination, Func<Weight, Weight, Weight> map)
    {
        NumericsDebug.AssertSameDimensions(matrices.a, matrices.b, destination);
        SpanOperations.MapTo(matrices.a.AsSpan(), matrices.b.AsSpan(), destination.AsSpan(), map);
    }
    public static Matrix Map(this (Matrix a, Matrix b, Matrix c) matrices, Func<Weight, Weight, Weight, Weight> map)
    {
        var destination = Matrix.OfSize(matrices.a);
        matrices.MapTo(destination, map);
        return destination;
    }
    public static void MapTo(this (Matrix a, Matrix b, Matrix c) matrices, Matrix destination, Func<Weight, Weight, Weight, Weight> map)
    {
        NumericsDebug.AssertSameDimensions(matrices.a, matrices.b, matrices.c, destination);
        SpanOperations.MapTo(matrices.a.AsSpan(), matrices.b.AsSpan(), matrices.c.AsSpan(), destination.AsSpan(), map);
    }

    [GenerateVariants]
    public static void MultiplyTo(this Matrix vector, Weight factor, Matrix destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        TensorPrimitives.Multiply(vector.AsSpan(), factor, destination.AsSpan());
    }

    [GenerateVariants]
    public static void DivideTo(this Matrix vector, Weight divisor, Matrix destination)
    {
        MultiplyTo(vector, 1 / divisor, destination);
    }

    public static Span<Weight> RowSpan(this Matrix matrix, int rowIndex) => matrix.AsSpan().Slice(rowIndex * matrix.ColumnCount, matrix.ColumnCount);
    public static Vector RowRef(this Matrix matrix, int rowIndex) => matrix.Storage.Slice(matrix.ColumnCount * rowIndex, matrix.ColumnCount);
    public static Matrix Rows(this Matrix matrix, Range range)
    {
        var (offset, length) = range.GetOffsetAndLength(matrix.RowCount);
        if (offset is 0 && length == matrix.RowCount)
        {
            return matrix;
        }
        return Matrix.Of(length, matrix.ColumnCount, matrix.Storage.Slice(offset * matrix.ColumnCount, length * matrix.ColumnCount));
    }

    public static Matrix CreateCopy(this Matrix matrix)
    {
        var copy = Matrix.Create(matrix.RowCount, matrix.ColumnCount);
        matrix.AsSpan().CopyTo(copy.AsSpan());
        return copy;
    }
    public static void CopyTo(this Matrix matrix, Matrix destination)
    {
        Debug.Assert(matrix.ColumnCount == destination.ColumnCount);
        Debug.Assert(matrix.RowCount <= destination.RowCount);
        matrix.AsSpan().CopyTo(destination.AsSpan());
    }

    public static void Fill(this Matrix matrix, Weight value) => matrix.AsSpan().Fill(value);
    public static void ResetZero(this Matrix matrix) => matrix.AsSpan().Clear();
}