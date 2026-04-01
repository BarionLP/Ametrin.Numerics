using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Threading;

namespace Ametrin.Numerics;

// owns the memory
public sealed class ArrayHandle(Weight[]? array, ArrayPool<Weight>? pool) : IDisposable
{
    public static ArrayHandle Disposed { get; } = new(null, null);
    private readonly ArrayPool<Weight>? pool = pool;
    private Weight[]? array = array;
    public Weight[]? Array => array;
    public int Length => array?.Length ?? 0;

    [MemberNotNullWhen(false, nameof(array)), MemberNotNullWhen(false, nameof(Array))]
    public bool IsDisposed => array is null;

    public Span<Weight> AsSpan()
    {
        Debug.Assert(!IsDisposed);
        return array;
    }

    public Span<Weight> AsSpan(int start, int length)
    {
        Debug.Assert(!IsDisposed);
        return array.AsSpan(start, length);
    }

    public Span<Weight> AsSpan(Range range)
    {
        Debug.Assert(!IsDisposed);
        return array.AsSpan(range);
    }

    public void Dispose()
    {
        // set storage to null and return old storage atomically
        var local = Interlocked.Exchange(ref array, null);
        if (local is null) return; // if local is null a different thread beat us to the disposal

        pool?.Return(local);
#if DEBUG
        GC.SuppressFinalize(this);
#endif
    }

#if DEBUG
    ~ArrayHandle()
    {
        if (array is null || pool is null) return;
        Console.WriteLine("not disposed handle");
    }
#endif
}
