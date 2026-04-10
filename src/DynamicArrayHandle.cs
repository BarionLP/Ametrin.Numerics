using System.Buffers;

namespace Ametrin.Numerics;

public sealed class DynamicArrayHandle(ArrayPool<Weight> pool) : IDisposable
{
    public DynamicArrayHandle() : this(ArrayPool<Weight>.Shared) { }
    public ArrayHandle Handle { get; private set; } = new(null, pool);
    public ArrayPool<float> Pool => Handle.Pool!;

    public void SetCount(int newCount)
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


// [Obsolete]
public sealed class Dynamic<TTensor>(ArrayPool<Weight> pool) : IDisposable
    where TTensor : struct, ITensorLike<TTensor>
{
    public Dynamic() : this(ArrayPool<Weight>.Shared) { }
    internal readonly DynamicArrayHandle handle = new(pool);
    public ArrayPool<float> Pool => handle.Pool;
    public TTensor Tensor { get; internal set; } = TTensor.Empty;

    public void OfSize(TTensor template)
    {
        handle.SetCount(template.FlatCount);

        Tensor = TTensor.OfSize(template, handle.Handle);
    }

    public void Dispose()
    {
        handle.Dispose();
        Tensor = TTensor.Empty;
    }
}

public static class DynamicTensorExtensions
{
    extension(Dynamic<Vector> dynamic)
    {
        public void SetCount(int count)
        {
            dynamic.handle.SetCount(count);
            dynamic.Tensor = Vector.Of(count, dynamic.handle.Handle);
        }
    }

    extension(Dynamic<Matrix> dynamic)
    {
        public void SetCount(int rowCount, int columnCount)
        {
            dynamic.handle.SetCount(rowCount * columnCount);
            dynamic.Tensor = Matrix.Of(rowCount, columnCount, dynamic.handle.Handle);
        }
    }

    extension(Dynamic<Tensor> dynamic)
    {
        public void SetCount(int rowCount, int columnCount, int layerCount)
        {
            dynamic.handle.SetCount(rowCount * columnCount * layerCount);
            dynamic.Tensor = Tensor.Of(rowCount, columnCount, layerCount, dynamic.handle.Handle);
        }
    }
}