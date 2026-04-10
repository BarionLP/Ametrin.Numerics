using System.Buffers;

namespace Ametrin.Numerics;

public sealed class DynamicArrayHandle(ArrayPool<Weight> pool) : IDisposable
{
    public DynamicArrayHandle() : this(ArrayPool<Weight>.Shared) { }
    public ArrayHandle Handle { get; private set; } = new(null, pool);
    public ArrayPool<float> Pool => Handle.Pool!;

    public void SetMinCapacity(int newCount)
    {
        Debug.Assert(newCount >= 0);
        if (Handle.IsDisposed || Handle.Length < newCount)
        {
            Dispose();
            Handle = Pool.RentNumerics(newCount);
        }
    }

    public void Dispose()
    {
        Handle.Dispose();
    }
}

public sealed class Dynamic<TTensor>(ArrayPool<Weight> pool) : IDisposable
    where TTensor : struct, ITensorLike<TTensor>
{
    public Dynamic() : this(ArrayPool<Weight>.Shared) { }
    internal readonly DynamicArrayHandle handle = new(pool);
    public ArrayPool<float> Pool => handle.Pool;
    public TTensor Tensor = TTensor.Empty;
    public Span<Weight> AsSpan() => Tensor.AsSpan();

    public void SetSize(TTensor template)
    {
        if (!TTensor.HaveSameSize(template, Tensor))
        {
            handle.SetMinCapacity(template.FlatCount);
            Tensor = TTensor.OfSize(template, handle.Handle);
        }
    }

    public void Dispose()
    {
        handle.Dispose();
        Tensor = TTensor.Empty;
    }

    public static implicit operator TTensor(Dynamic<TTensor> dynamic) => dynamic.Tensor;
}

public static class DynamicTensorExtensions
{
    extension(Dynamic<Vector> dynamic)
    {
        public void SetSize(int count)
        {
            if(dynamic.Tensor.Count != count)
            {
                dynamic.handle.SetMinCapacity(count);
                dynamic.Tensor = Vector.Of(count, dynamic.handle.Handle);
            }
        }
    }

    extension(Dynamic<Matrix> dynamic)
    {
        public void SetSize(int rowCount, int columnCount)
        {
            if (dynamic.Tensor.RowCount != rowCount || dynamic.Tensor.ColumnCount != columnCount)
            {
                dynamic.handle.SetMinCapacity(rowCount * columnCount);
                dynamic.Tensor = Matrix.Of(rowCount, columnCount, dynamic.handle.Handle);
            }
        }
    }

    extension(Dynamic<Tensor> dynamic)
    {
        public void SetSize(int rowCount, int columnCount, int layerCount)
        {
            dynamic.handle.SetMinCapacity(rowCount * columnCount * layerCount);
            dynamic.Tensor = Tensor.Of(rowCount, columnCount, layerCount, dynamic.handle.Handle);
        }
    }
}