using System.Buffers;

namespace Ametrin.Numerics;

public static class NumericsArrayPoolExtensions
{
    extension(ArrayPool<Weight> pool)
    {
        public ArrayHandle RentNumerics(int minSize, bool cleared = false)
        {
            var storage = pool.Rent(minSize);
            if (cleared) storage.AsSpan().Clear();
            return new(storage, pool);
        }
    }
}
