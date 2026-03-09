using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Threading;

namespace Ametrin.Numerics;

public static class NumericsArrayPoolExtensions
{
    extension(ArrayPool<Weight> pool)
    {
        public StorageHandle RentNumerics(int minSize, bool cleared = false)
        {
            var storage = pool.Rent(minSize);
            if (cleared) storage.AsSpan().Clear();
            return new(storage, pool);
        }
    }
}

// owns the memory
public sealed class StorageHandle(Weight[]? storage, ArrayPool<Weight> pool) : IDisposable
{
    public static StorageHandle Disposed { get; } = new(null, ArrayPool<float>.Shared);
    internal readonly ArrayPool<Weight> pool = pool;
    internal Weight[]? storage = storage;
    [MemberNotNullWhen(false, nameof(storage))]
    public bool IsDisposed => storage is null;

    public ref Weight this[int index] => ref storage![index];

    public ref Weight this[nuint index] => ref storage![(int)index];
    public int Length => storage!.Length;

    public Span<Weight> AsSpan()
    {
        Debug.Assert(!IsDisposed);
        return storage;
    }

    public Span<Weight> AsSpan(Range range)
    {
        Debug.Assert(!IsDisposed);
        return storage.AsSpan(range);
    }


    public void Dispose()
    {
        // set storage to null and return old storage atomically  
        var local = Interlocked.Exchange(ref storage, null);
        if (local is null) return; // if local is null a different thread beat us to the disposal

        pool.Return(local);
#if DEBUG
        GC.SuppressFinalize(this);
#endif
    }

#if DEBUG
    ~StorageHandle()
    {
        if (storage is null) return;
        Console.WriteLine("not disposed handle");
    }
#endif
}

public struct DynamicVector(ArrayPool<Weight> pool) : IDisposable
{
    public DynamicVector() : this(ArrayPool<Weight>.Shared) { }
    private readonly ArrayPool<Weight> pool = pool;
    private StorageHandle handle = StorageHandle.Disposed;
    public Vector Vector { get; private set; } = Vector.Empty;

    public void SetCount(int newCount)
    {
        Debug.Assert(newCount >= 0);
        if (handle.IsDisposed || handle.Length < newCount)
        {
            Dispose();
            handle = pool.RentNumerics(newCount);
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
