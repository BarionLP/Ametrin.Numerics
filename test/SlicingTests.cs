namespace Ametrin.Numerics.Test;

public sealed class SlicingTests
{
    [Test]
    public async Task Vector_Slice()
    {
        var vector = Vector.Of([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        var slice = vector.Slice(2, 5);
        await Assert.That(slice.AsSpan().SequenceEqual([2, 3, 4, 5, 6])).IsTrue();
        var sliceOfSlice = slice.Slice(1, 3);
        await Assert.That(sliceOfSlice.AsSpan().SequenceEqual([3, 4, 5])).IsTrue();
    }
}
