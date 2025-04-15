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

    [Test]
    public async Task Matrix_Rows()
    {
        var matrix = Matrix.Of(5, 2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        await Assert.That(matrix.Rows(..5).Storage).IsEqualTo(matrix.Storage);
        var slice = matrix.Rows(1..4);
        await Assert.That(slice.AsSpan().SequenceEqual([2, 3, 4, 5, 6, 7])).IsTrue();
        var sliceOfSlice = slice.Rows(2..3);
        await Assert.That(sliceOfSlice.AsSpan().SequenceEqual([6, 7])).IsTrue();
        await Assert.That(matrix.RowRef(3).AsSpan().SequenceEqual([6, 7])).IsTrue();
        await Assert.That(matrix.RowSpan(2).SequenceEqual([4, 5])).IsTrue();
    }
}
