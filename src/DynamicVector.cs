using System.Buffers;

namespace Ametrin.Numerics;

public sealed class DynamicVector(ArrayPool<Weight> pool) : IDisposable
{
    public DynamicVector() : this(ArrayPool<Weight>.Shared) { }
    private ArrayHandle handle = ArrayHandle.Disposed;
    public ArrayPool<float> Pool { get; } = pool;
    public Vector Vector { get; private set; } = Vector.Empty;

    public void SetCount(int newCount)
    {
        Debug.Assert(newCount >= 0);
        if (handle.IsDisposed || handle.Length < newCount)
        {
            Dispose();
            handle = Pool.RentNumerics(newCount);
        }

        if (Vector.Count != newCount)
        {
            Vector = Vector.Of(handle, newCount);
        }
    }


    public void Dispose()
    {
        handle.Dispose();
        Vector = Vector.Empty;
    }
}
