namespace Ametrin.Numerics;

public interface Tensor
{
    public int RowCount { get; }
    public int ColumnCount { get; }
    public int LayerCount { get; }
    public int FlatCount { get; }
    public ref Weight this[int row, int column, int layer] { get; }
    public ref Weight this[nuint flatIndex] { get; }
    public Vector Storage { get; }

    public Span<Weight> AsSpan();

    public static Tensor CreateCube(int size) => Create(size, size, size);
    public static Tensor Create(int rowCount, int columnCount, int layerCount) => new TensorFlat(rowCount, columnCount, layerCount, Vector.Create(rowCount * columnCount * layerCount));
    public static Tensor Of(int rowCount, int columnCount, int layerCount, Weight[] storage) => Of(rowCount, columnCount, layerCount, Vector.Of(storage));
    public static Tensor Of(int rowCount, int columnCount, int layerCount, Vector storage)
    {
        if (storage.Count != rowCount * columnCount * layerCount)
        {
            throw new ArgumentException("storage size does not match specified dimensions");
        }

        return new TensorFlat(rowCount, columnCount, layerCount, storage);
    }

    public static Tensor OfSize(Tensor template) => Create(template.RowCount, template.ColumnCount, template.LayerCount);
}

public readonly struct TensorFlat(int rowCount, int columnCount, int layerCount, Vector storage) : Tensor
{
    public ref Weight this[int row, int column, int layer] => ref Storage[GetFlatIndex(row, column, layer)];
    public ref Weight this[nuint flatIndex] => ref Storage[flatIndex];

    public int RowCount { get; } = rowCount;
    public int ColumnCount { get; } = columnCount;
    public int LayerCount { get; } = layerCount;

    public int FlatCount => Storage.Count;

    public Vector Storage { get; } = storage;

    public Span<Weight> AsSpan() => Storage.AsSpan();

    internal int GetFlatIndex(int row, int column, int layer)
    {
#if DEBUG
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, RowCount);
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, ColumnCount);
        ArgumentOutOfRangeException.ThrowIfNegative(column);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layer, LayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
#endif

        return layer * RowCount * ColumnCount + row * ColumnCount + column;
    }
}

[NumericsHelper<Tensor>]
public static partial class TensorHelper
{
    [GenerateVariants]
    public static void MapTo(this Tensor tensor, Func<Weight, Weight> map, Tensor destination)
    {
        NumericsDebug.AssertSameDimensions(tensor, destination);
        SpanOperations.MapTo(tensor.AsSpan(), destination.AsSpan(), map);
    }
    public static Tensor Map(this (Tensor a, Tensor b) tensors, Func<Weight, Weight, Weight> map)
    {
        var destination = Tensor.OfSize(tensors.a);
        SpanOperations.MapTo(tensors.a.AsSpan(), tensors.b.AsSpan(), destination.AsSpan(), map);
        return destination;
    }
    public static void MapToFirst(this (Tensor a, Tensor b) tensors, Func<Weight, Weight, Weight> map)
        => SpanOperations.MapTo(tensors.a.AsSpan(), tensors.b.AsSpan(), tensors.a.AsSpan(), map);

    [GenerateVariants]
    public static void AddTo(this Tensor left, Tensor right, Tensor destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        TensorPrimitives.Add(left.AsSpan(), right.AsSpan(), destination.AsSpan());
    }

    [GenerateVariants]
    public static void PointwiseMultiplyTo(this Tensor left, Tensor right, Tensor destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        TensorPrimitives.Multiply(left.AsSpan(), right.AsSpan(), destination.AsSpan());
    }

    [GenerateVariants]
    public static void SubtractTo(this Tensor left, Tensor right, Tensor destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        TensorPrimitives.Subtract(left.AsSpan(), right.AsSpan(), destination.AsSpan());
    }

    public static Matrix LayerRef(this Tensor tensor, int layer) => new TensorLayerReference(layer, tensor);

    public static Tensor CreateCopy(this Tensor tensor)
    {
        var copy = Tensor.Create(tensor.RowCount, tensor.ColumnCount, tensor.LayerCount);
        tensor.CopyTo(copy);
        return copy;
    }

    public static void CopyTo(this Tensor tensor, Tensor destination)
    {
        NumericsDebug.AssertSameDimensions(tensor, destination);
        tensor.AsSpan().CopyTo(destination.AsSpan());
    }

    public static void Fill(this Tensor tensor, Weight value) => tensor.AsSpan().Fill(value);
    public static void ResetZero(this Tensor tensor) => tensor.AsSpan().Clear();
}