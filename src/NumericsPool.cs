// using System.Buffers;

// namespace Ametrin.Numerics;


// public sealed class StorePool(ArrayPool<float>? pool = null) : IDisposable
// {
//     private readonly ArrayPool<float> pool = pool ?? ArrayPool<float>.Shared;
//     private readonly Stack<StoreHandle> handleCache = [];

//     public StoreHandle Rent(int minLength)
//     {
//         var storage = pool.Rent(minLength);

//         if (handleCache.TryPop(out var handle))
//         {
//             handle.storage = storage;
//             return handle;
//         }
//         else
//         {
//             return new(storage, this);
//         }
//     }

//     public void Return(StoreHandle handle)
//     {
//         Debug.Assert(handle.pool == this);
//         Debug.Assert(handle.storage is not null);
//         pool.Return(handle.storage);

//         handle.storage = null;
//     }

//     public void Dispose()
//     {
//         handleCache.Clear();
//     }
// }

// public sealed class StoreHandle(Weight[] storage, StorePool? pool = null) : IDisposable
// {
//     internal readonly StorePool? pool = pool;
//     internal Weight[]? storage = storage is null ? throw new ArgumentNullException(nameof(storage)) : storage;
//     public bool IsDisposed => storage is null;

//     public ref Weight this[int index] => ref storage![index];

//     public ref Weight this[nuint index] => ref storage![(int)index];

//     public Span<Weight> AsSpan()
//     {
//         Debug.Assert(!IsDisposed);
//         return storage;
//     }

//     public Span<Weight> AsSpan(Range range)
//     {
//         Debug.Assert(!IsDisposed);
//         return storage.AsSpan(range);
//     }

//     public Memory<Weight> AsMemory() => storage.AsMemory();
//     public Memory<Weight> AsMemory(Range range) => storage.AsMemory(range);


//     public void Dispose()
//     {
//         pool?.Return(this);
//         storage = null!;
// #if DEBUG
//         GC.SuppressFinalize(this);
// #endif
//     }

// #if DEBUG
//     ~StoreHandle()
//     {
//         if (pool is null) return;
//         Console.WriteLine("not disposed handle");
//     }
// #endif
// }
